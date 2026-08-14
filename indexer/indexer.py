# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Video Indexer — Indexing Job
# MAGIC
# MAGIC Pipeline de indexación de un video (invocado por la Databricks App vía Jobs API).
# MAGIC
# MAGIC | Paso | Descripción |
# MAGIC |------|-------------|
# MAGIC | 1 | Extraer audio con ffmpeg (WAV mono 16 kHz) |
# MAGIC | 2 | Transcribir con Whisper (segmentos + timestamps + idioma) |
# MAGIC | 3 | Extraer frames del video (1 cada `FRAME_INTERVAL_S` s) |
# MAGIC | 4 | OCR + descripción visual de cada frame con `ai_parse_document` |
# MAGIC | 5 | Enriquecer con LLM: resumen + temas, entidades, capítulos |
# MAGIC | 6 | Generar embeddings (databricks-gte-large-en, 1024 dims) |
# MAGIC | 7 | Escribir todo en Lakebase (pgvector) |
# MAGIC
# MAGIC Parámetros vía widgets (los pasa la App). El estado del video se actualiza en
# MAGIC la tabla `videos` (`indexing` → `indexed` / `error`).

# COMMAND ----------

# DBTITLE 1,1 — Dependencias (según CPU/GPU)
# En GPU (runtime ML) usamos openai-whisper (torch/cuda, ya preinstalado y estable).
# En CPU usamos faster-whisper (CTranslate2 int8): ligero y sin torch.
import subprocess, sys
_gpu_env = False
try:
    import torch
    _gpu_env = torch.cuda.is_available()
except Exception:
    _gpu_env = False
_pkgs = ["databricks-sdk>=0.118.0", "imageio-ffmpeg", "psycopg[binary]>=3.1.0"]
_pkgs += ["openai-whisper"] if _gpu_env else ["faster-whisper>=1.0.0"]
print("GPU detectada → openai-whisper" if _gpu_env else "CPU → faster-whisper", _pkgs)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-q", *_pkgs])
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,2 — Parámetros
dbutils.widgets.text("video_id",        "",                                   "video_id")
dbutils.widgets.text("volume_path",     "",                                   "Ruta del video en el Volume")
dbutils.widgets.text("frames_volume",   "/Volumes/jgworkspaceclassic_catalog/video_indexer/frames", "Volume de frames")
dbutils.widgets.text("whisper_model",   "small",                              "Modelo Whisper")
dbutils.widgets.text("frame_interval",  "5",                                  "Segundos entre frames")
dbutils.widgets.text("language",        "auto",                               "Idioma (auto|es|en|...)")

dbutils.widgets.text("lb_project",      "video-indexer",                      "Lakebase project")
dbutils.widgets.text("lb_branch",       "production",                         "Lakebase branch")
dbutils.widgets.text("lb_endpoint",     "primary",                            "Lakebase endpoint")
dbutils.widgets.text("lb_host",         "ep-polished-heart-d2wgybwr.database.us-east-1.cloud.databricks.com", "Lakebase host")
dbutils.widgets.text("lb_db",           "databricks_postgres",                "Lakebase database")

dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en",         "Embedding endpoint")
dbutils.widgets.text("llm_endpoint",       "databricks-claude-sonnet-5",      "LLM endpoint")
# OCR de frames corre en un SQL warehouse serverless (ai_parse_document no está
# disponible en el Spark del cluster clásico donde corre Whisper).
dbutils.widgets.text("warehouse_id",       "740f0a066259b349",                "SQL Warehouse (OCR)")

VIDEO_ID      = dbutils.widgets.get("video_id").strip()
VOLUME_PATH   = dbutils.widgets.get("volume_path").strip()
FRAMES_VOLUME = dbutils.widgets.get("frames_volume").strip().rstrip("/")
WHISPER_MODEL = dbutils.widgets.get("whisper_model").strip() or "small"
FRAME_INTERVAL_S = int(dbutils.widgets.get("frame_interval").strip() or "5")
LANGUAGE      = dbutils.widgets.get("language").strip() or "auto"

LB_PROJECT    = dbutils.widgets.get("lb_project").strip()
LB_BRANCH     = dbutils.widgets.get("lb_branch").strip()
LB_ENDPOINT   = dbutils.widgets.get("lb_endpoint").strip()
LB_HOST       = dbutils.widgets.get("lb_host").strip()
LB_DB         = dbutils.widgets.get("lb_db").strip()

EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint").strip()
LLM_ENDPOINT       = dbutils.widgets.get("llm_endpoint").strip()
WAREHOUSE_ID       = dbutils.widgets.get("warehouse_id").strip()
EMBEDDING_DIM      = 1024

# Filtros de OCR de frames
MIN_FRAME_CHARS = 12          # texto OCR mínimo por frame para indexarlo

assert VIDEO_ID,    "video_id es obligatorio"
assert VOLUME_PATH, "volume_path es obligatorio"

AUDIO_TMP = f"/tmp/vi_{VIDEO_ID}.wav"
LB_ENDPOINT_NAME = f"projects/{LB_PROJECT}/branches/{LB_BRANCH}/endpoints/{LB_ENDPOINT}"

print(f"video_id     : {VIDEO_ID}")
print(f"video        : {VOLUME_PATH}")
print(f"whisper      : {WHISPER_MODEL} | idioma: {LANGUAGE} | frame cada {FRAME_INTERVAL_S}s")
print(f"lakebase     : {LB_HOST} / {LB_DB} ({LB_ENDPOINT_NAME})")
print(f"embeddings   : {EMBEDDING_ENDPOINT} ({EMBEDDING_DIM}) | llm: {LLM_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,3 — Helpers de Lakebase
import psycopg
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
USERNAME = w.current_user.me().user_name

def lb_conn():
    """Nueva conexión Postgres con token OAuth fresco (válido ~1h)."""
    token = w.postgres.generate_database_credential(endpoint=LB_ENDPOINT_NAME).token
    return psycopg.connect(
        host=LB_HOST, dbname=LB_DB, user=USERNAME, password=token, sslmode="require"
    )

def set_status(status, **fields):
    """Actualiza la fila del video en la tabla `videos`."""
    cols = ["status = %s"]
    vals = [status]
    for k, v in fields.items():
        cols.append(f"{k} = %s")
        vals.append(v)
    vals.append(VIDEO_ID)
    with lb_conn() as c, c.cursor() as cur:
        cur.execute(f"UPDATE videos SET {', '.join(cols)} WHERE video_id = %s", vals)
        c.commit()

# Marca inicio del proceso
try:
    set_status("indexing", error_msg=None)
    print("Estado -> indexing")
except Exception as e:
    print(f"No se pudo marcar 'indexing' (¿fila creada por la app?): {e}")

# COMMAND ----------

# DBTITLE 1,4 — Extraer audio
import subprocess, os, imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

try:
    cmd = [FFMPEG, "-y", "-i", VOLUME_PATH, "-ac", "1", "-ar", "16000", "-vn", AUDIO_TMP]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg audio error:\n{r.stderr[-800:]}")
    print(f"Audio: {AUDIO_TMP} ({os.path.getsize(AUDIO_TMP)/1e6:.1f} MB)")
except Exception as e:
    set_status("error", error_msg=f"audio: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,5 — Transcripción (GPU: openai-whisper / CPU: faster-whisper)
import os, wave, numpy as np

# Cargar el WAV como array float32 (evita que whisper invoque ffmpeg del sistema)
with wave.open(AUDIO_TMP, "rb") as wf:
    _raw = wf.readframes(wf.getnframes()); _sr = wf.getframerate()
audio_np = np.frombuffer(_raw, dtype=np.int16).astype(np.float32) / 32768.0
duration_s = round(len(audio_np) / _sr, 1)

_gpu = False
try:
    import torch
    _gpu = torch.cuda.is_available()
except Exception:
    _gpu = False

try:
    if _gpu:
        import whisper
        print(f"Transcribiendo en GPU (openai-whisper '{WHISPER_MODEL}')...")
        wmodel = whisper.load_model(WHISPER_MODEL, device="cuda")
        kw = dict(verbose=False, fp16=True)
        if LANGUAGE != "auto":
            kw["language"] = LANGUAGE
        tr = wmodel.transcribe(audio_np, **kw)
        detected_lang = tr.get("language", LANGUAGE)
        segments = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
                    for s in tr["segments"] if s["text"].strip()]
    else:
        # Estabilidad de librerías nativas en CPU (fijar ANTES de importar ctranslate2)
        os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("CT2_USE_MKL", "0")
        from faster_whisper import WhisperModel
        print(f"Transcribiendo en CPU (faster-whisper '{WHISPER_MODEL}' int8)...")
        fmodel = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                              cpu_threads=1, num_workers=1)
        seg_iter, info = fmodel.transcribe(
            audio_np, language=None if LANGUAGE == "auto" else LANGUAGE,
            beam_size=5, vad_filter=True)
        detected_lang = info.language
        segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                    for s in seg_iter if s.text.strip()]

    full_text = " ".join(s["text"] for s in segments)
    print(f"Duración: {duration_s}s | idioma: {detected_lang} | "
          f"segmentos: {len(segments)} | GPU: {_gpu}")
except Exception as e:
    set_status("error", error_msg=f"transcripcion: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,6 — Extraer frames
import shutil

vclean = VIDEO_ID.replace("-", "")
frames_dir = f"{FRAMES_VOLUME}/{vclean}"
try:
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [FFMPEG, "-y", "-i", VOLUME_PATH,
           "-vf", f"fps=1/{FRAME_INTERVAL_S}", "-q:v", "3",
           f"{frames_dir}/frame_%04d.jpg"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg frames error:\n{r.stderr[-800:]}")
    n_extracted = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    print(f"Frames extraídos: {n_extracted} -> {frames_dir}")

    # --- Dedup VISUAL: cuando la misma imagen dura varios segundos, ffmpeg genera
    # frames idénticos. Comparamos un perceptual hash (aHash 8x8) contra el último
    # frame conservado y borramos los visualmente iguales ANTES del OCR.
    # Se conservan los nombres de archivo → el timestamp por frame sigue siendo correcto.
    AHASH_MAXDIST = 5   # distancia de Hamming (de 64 bits) para considerar "igual"
    try:
        from PIL import Image
        import numpy as _np

        def _ahash(fp, s=8):
            a = _np.asarray(Image.open(fp).convert("L").resize((s, s)), dtype=_np.float32)
            return (a > a.mean()).flatten()

        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        prev_h, removed = None, 0
        for f in files:
            fp = os.path.join(frames_dir, f)
            try:
                h = _ahash(fp)
            except Exception:
                continue
            if prev_h is not None and int(_np.count_nonzero(h != prev_h)) <= AHASH_MAXDIST:
                os.remove(fp); removed += 1      # visualmente idéntico al anterior → fuera
            else:
                prev_h = h
        print(f"Dedup visual: eliminados {removed} frames repetidos")
    except Exception as e:
        print(f"(dedup visual omitido: {e})")

    n_frames_files = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    print(f"Frames únicos para OCR: {n_frames_files}")
except Exception as e:
    # Los frames no son críticos: seguimos sin OCR si fallan
    print(f"⚠️ Falló extracción de frames: {e}")
    n_frames_files = 0

# COMMAND ----------

# DBTITLE 1,7 — OCR de frames con ai_parse_document (vía SQL warehouse serverless)
# ai_parse_document NO está disponible en el Spark del cluster clásico → ejecutamos
# el OCR contra un SQL warehouse serverless usando la Statement Execution API.
import time as _time
from databricks.sdk.service.sql import StatementState

frame_rows = []   # (start_time, end_time, text) tras colapsar duplicados
if n_frames_files > 0 and WAREHOUSE_ID:
    try:
        ocr_sql = f"""
            SELECT CAST(regexp_extract(path,'frame_([0-9]+)',1) AS INT) AS fnum,
                   concat_ws(' ', collect_list(t)) AS txt
            FROM (
              SELECT path,
                     -- ai_parse_document devuelve tablas como HTML; quitamos las etiquetas.
                     -- OJO: NO usar '\\s+' aquí: Spark SQL se come el backslash y el patrón
                     -- queda como 's+' (borra las 's'). La normalización de espacios se hace
                     -- en Python (abajo) con str.split(), que es seguro.
                     regexp_replace(
                       COALESCE(try_cast(el:content AS STRING), try_cast(el:description AS STRING)),
                       '<[^>]+>', ' ') AS t
              FROM (
                SELECT path,
                       ai_parse_document(content, map('version','2.0','descriptionElementTypes','*')) AS parsed
                FROM READ_FILES('{frames_dir}', format => 'binaryFile')
              )
              LATERAL VIEW EXPLODE(from_json(to_json(parsed:document:elements),'ARRAY<VARIANT>')) AS el
              WHERE COALESCE(try_cast(el:content AS STRING), try_cast(el:description AS STRING)) IS NOT NULL
            )
            GROUP BY fnum
            ORDER BY fnum
        """
        stmt = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID, statement=ocr_sql, wait_timeout="50s")
        sid = stmt.statement_id
        # Poll hasta terminar (el OCR de muchos frames puede tardar > 50s)
        while stmt.status.state in (StatementState.PENDING, StatementState.RUNNING):
            _time.sleep(5)
            stmt = w.statement_execution.get_statement(sid)
        if stmt.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f"OCR SQL {stmt.status.state}: "
                               f"{getattr(stmt.status, 'error', None)}")

        # Recolectar todos los chunks de resultados
        data = list(stmt.result.data_array or [])
        nxt = stmt.result.next_chunk_index if stmt.result else None
        while nxt is not None:
            ch = w.statement_execution.get_statement_result_chunk_n(sid, nxt)
            data.extend(ch.data_array or [])
            nxt = ch.next_chunk_index

        # Colapsar corridas de frames con el MISMO contenido en pantalla:
        # cuando una imagen dura varios segundos genera frames repetidos. Comparamos
        # el texto por similitud de tokens (Jaccard) contra el frame previo conservado;
        # si es casi igual, fusionamos extendiendo el rango de tiempo en vez de duplicar.
        # str.split() colapsa cualquier whitespace (espacios/tabs/saltos) de forma segura
        rows_sorted = sorted(
            [(int(r[0]) if r[0] is not None else 0, " ".join((r[1] or "").split())) for r in data],
            key=lambda x: x[0])

        SIM_THRESHOLD = 0.80  # >= => se considera el mismo frame
        kept = []             # dicts: start, end, text, toks
        for fnum, txt in rows_sorted:
            if len(txt) < MIN_FRAME_CHARS:
                continue
            ts = (fnum - 1) * FRAME_INTERVAL_S
            tk = set(txt.lower().split())
            if kept:
                prev = kept[-1]
                union = len(tk | prev["toks"]) or 1
                if len(tk & prev["toks"]) / union >= SIM_THRESHOLD:
                    prev["end"] = ts + FRAME_INTERVAL_S           # extiende el rango
                    if len(txt) > len(prev["text"]):              # conserva el texto más completo
                        prev["text"], prev["toks"] = txt, tk
                    continue
            kept.append({"start": float(ts), "end": float(ts + FRAME_INTERVAL_S),
                         "text": txt, "toks": tk})
        frame_rows = [(k["start"], k["end"], k["text"]) for k in kept]
        print(f"Frames con OCR útil: {len(frame_rows)} "
              f"(colapsados de {len(rows_sorted)} frames con texto)")
    except Exception as e:
        print(f"⚠️ Falló OCR de frames: {e}")
else:
    print("OCR omitido (sin frames o sin warehouse configurado).")

# COMMAND ----------

# DBTITLE 1,8 — Enriquecimiento con LLM (resumen, temas, entidades, capítulos)
import json, re
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

def _content_text(msg) -> str:
    """Extrae texto de la respuesta: los modelos de razonamiento devuelven
    content como lista de bloques en vez de string."""
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict):
                parts.append(b.get("text", "") or "")
            elif isinstance(b, str):
                parts.append(b)
            else:
                parts.append(getattr(b, "text", "") or "")
        return "".join(parts)
    return str(c or "")


def llm_json(prompt, max_tokens=1200):
    # nota: los modelos de razonamiento (claude-sonnet-5) no aceptan 'temperature'
    resp = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=max_tokens,
    )
    raw = _content_text(resp.choices[0].message).strip()
    m = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    try:
        return json.loads(m.group() if m else raw)
    except Exception:
        return None

# Transcripción compacta con timestamps para dar contexto temporal al LLM
ts_lines = [f"[{int(s['start'])}s] {s['text'].strip()}" for s in segments]
transcript_ts = "\n".join(ts_lines)[:12000]
transcript_plain = full_text[:8000]

description, topics, entities, chapters = "", [], [], []

# 8a. Resumen + temas
meta = llm_json(
    "Eres un analista de contenido de video. A partir de la transcripción, responde SOLO con un JSON:\n"
    '{"descripcion": "<resumen de 3-4 oraciones del tema, contenido y propósito del video>",\n'
    ' "temas": ["<2 a 5 temas clave, cortos, en minúsculas>"]}\n\n'
    f"Transcripción:\n{transcript_plain}\n\nJSON:"
)
if meta:
    description = meta.get("descripcion", "") or ""
    t = meta.get("temas", [])
    topics = [str(x).strip().lower() for x in t] if isinstance(t, list) else []
print(f"Resumen: {description[:120]}...")
print(f"Temas: {topics}")

# 8b. Entidades
ent = llm_json(
    "Extrae las entidades mencionadas en la transcripción del video. Responde SOLO con un JSON array:\n"
    '[{"tipo": "persona|organizacion|producto|lugar|marca|evento|otro", "valor": "<nombre>"}]\n'
    "No inventes; solo entidades presentes. Máximo 30.\n\n"
    f"Transcripción:\n{transcript_plain}\n\nJSON:"
)
if isinstance(ent, list):
    agg = {}
    for e in ent:
        if not isinstance(e, dict):
            continue
        val = str(e.get("valor", "")).strip()
        typ = str(e.get("tipo", "otro")).strip().lower()
        if not val:
            continue
        key = (typ, val.lower())
        if key not in agg:
            agg[key] = {"entity_type": typ, "entity_value": val, "mentions": 1}
        else:
            agg[key]["mentions"] += 1
    entities = list(agg.values())
print(f"Entidades: {len(entities)}")

# 8c. Capítulos / momentos clave
chap = llm_json(
    "Divide el video en capítulos temáticos usando los timestamps (en segundos) de la transcripción. "
    "Responde SOLO con un JSON array ordenado por tiempo:\n"
    '[{"inicio_s": <segundo de inicio>, "titulo": "<título corto>", "resumen": "<1 oración>"}]\n'
    "Entre 3 y 8 capítulos. El primero debe iniciar en 0.\n\n"
    f"Transcripción con timestamps:\n{transcript_ts}\n\nJSON:",
    max_tokens=1500,
)
if isinstance(chap, list):
    clean = []
    for i, c in enumerate(chap):
        if not isinstance(c, dict):
            continue
        try:
            start = float(c.get("inicio_s", 0))
        except Exception:
            start = 0.0
        clean.append({
            "chapter_idx": i,
            "start_time": start,
            "title": str(c.get("titulo", f"Capítulo {i+1}")).strip(),
            "summary": str(c.get("resumen", "")).strip(),
        })
    clean.sort(key=lambda x: x["start_time"])
    for i, c in enumerate(clean):
        c["chapter_idx"] = i
        c["end_time"] = clean[i + 1]["start_time"] if i + 1 < len(clean) else duration_s
    chapters = clean
print(f"Capítulos: {len(chapters)}")

# COMMAND ----------

# DBTITLE 1,9 — Construir chunks + embeddings
from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")

def embed_batch(texts):
    resp = deploy_client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": texts})
    return [d["embedding"] for d in resp["data"]]

# chunks: (source_type, seq, start_time, end_time, text)
chunks = []
for i, s in enumerate(segments):
    chunks.append(("transcript", i, float(s["start"]), float(s["end"]), s["text"].strip()))
for j, (st, et, txt) in enumerate(frame_rows):
    chunks.append(("ocr", j, st, et, txt))

print(f"Total chunks a vectorizar: {len(chunks)} "
      f"(transcript={len(segments)}, ocr={len(frame_rows)})")

embeddings = []
BATCH = 64
texts = [c[4] for c in chunks]
for i in range(0, len(texts), BATCH):
    embeddings.extend(embed_batch(texts[i:i + BATCH]))
    print(f"  embeddings {min(i+BATCH,len(texts))}/{len(texts)}", end="\r")
print(f"\nEmbeddings: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")

# COMMAND ----------

# DBTITLE 1,10 — Escribir en Lakebase
try:
    with lb_conn() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            # Idempotencia: limpiar datos previos de este video
            cur.execute("DELETE FROM video_chunks   WHERE video_id = %s", (VIDEO_ID,))
            cur.execute("DELETE FROM video_entities WHERE video_id = %s", (VIDEO_ID,))
            cur.execute("DELETE FROM video_chapters WHERE video_id = %s", (VIDEO_ID,))

            # chunks + embeddings
            ins = ("INSERT INTO video_chunks "
                   "(video_id, source_type, seq, start_time, end_time, text, embedding) "
                   "VALUES (%s,%s,%s,%s,%s,%s,%s::vector)")
            batch = []
            for (stype, seq, st, et, txt), emb in zip(chunks, embeddings):
                vec = "[" + ",".join(f"{v:.7f}" for v in emb) + "]"
                batch.append((VIDEO_ID, stype, seq, st, et, txt, vec))
                if len(batch) >= 100:
                    cur.executemany(ins, batch); batch = []
            if batch:
                cur.executemany(ins, batch)

            if entities:
                cur.executemany(
                    "INSERT INTO video_entities (video_id, entity_type, entity_value, mentions) "
                    "VALUES (%s,%s,%s,%s)",
                    [(VIDEO_ID, e["entity_type"], e["entity_value"], e["mentions"]) for e in entities],
                )
            if chapters:
                cur.executemany(
                    "INSERT INTO video_chapters (video_id, chapter_idx, start_time, end_time, title, summary) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    [(VIDEO_ID, c["chapter_idx"], c["start_time"], c["end_time"], c["title"], c["summary"])
                     for c in chapters],
                )

            cur.execute(
                "UPDATE videos SET status='indexed', duration_s=%s, language=%s, description=%s, "
                "topics=%s, n_segments=%s, n_frames=%s, error_msg=NULL, indexed_at=NOW() "
                "WHERE video_id=%s",
                (duration_s, detected_lang, description, topics, len(segments), len(frame_rows), VIDEO_ID),
            )
        conn.commit()
    print(f"✅ Indexado: {len(chunks)} chunks, {len(entities)} entidades, {len(chapters)} capítulos")
except Exception as e:
    set_status("error", error_msg=f"lakebase: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,11 — Limpieza de temporales
try:
    if os.path.exists(AUDIO_TMP):
        os.remove(AUDIO_TMP)
except Exception:
    pass
dbutils.notebook.exit(json.dumps({
    "video_id": VIDEO_ID, "status": "indexed",
    "segments": len(segments), "frames": len(frame_rows),
    "entities": len(entities), "chapters": len(chapters),
    "duration_s": duration_s, "language": detected_lang,
}))
