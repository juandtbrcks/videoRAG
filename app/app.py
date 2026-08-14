"""videoRAG — Databricks App (Streamlit).

Sube videos, indéxalos (transcripción + OCR + entidades + capítulos), búscalos
semánticamente y conversa con un agente RAG sobre todo el contenido indexado.
Datos en Lakebase (pgvector) · IA con Foundation Models.
"""
import uuid

import streamlit as st

import config
import storage
import agent
from db import get_db

st.set_page_config(page_title="videoRAG", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .badge { display:inline-block; padding:2px 10px; border-radius:12px;
               font-size:0.72rem; font-weight:600; }
      .b-indexed  { background:#e6f4ea; color:#137333; }
      .b-indexing { background:#fef7e0; color:#b06000; }
      .b-uploaded { background:#e8f0fe; color:#1a56db; }
      .b-error    { background:#fce8e6; color:#c5221f; }
      .card { background:#f7f8fa; border:1px solid #e5e8ec; border-radius:10px;
              padding:12px 14px; margin-bottom:8px; }
      .meta { font-size:0.75rem; color:#6b7280; }
      .snippet { font-size:0.85rem; color:#374151; margin-top:4px; }
      .score { background:#e8f0fe; color:#1a56db; border-radius:10px;
               padding:1px 8px; font-size:0.72rem; font-weight:600; }
      .topic { display:inline-block; background:#eef2ff; color:#3730a3;
               border-radius:12px; padding:2px 10px; font-size:0.75rem; margin:2px; }

      /* --- Menú de navegación estilo workspace --- */
      section[data-testid="stSidebar"] button[kind="tertiary"],
      section[data-testid="stSidebar"] button[kind="secondary"] {
          width:100%; justify-content:flex-start !important;
          padding:6px 12px !important; border:none !important; box-shadow:none !important;
          border-radius:8px !important; color:inherit !important;
      }
      section[data-testid="stSidebar"] button[kind="tertiary"] div,
      section[data-testid="stSidebar"] button[kind="secondary"] div,
      section[data-testid="stSidebar"] button[kind="tertiary"] p,
      section[data-testid="stSidebar"] button[kind="secondary"] p {
          text-align:left !important; justify-content:flex-start !important;
          width:100%; margin:0; font-size:0.95rem;
      }
      section[data-testid="stSidebar"] button[kind="tertiary"] {
          background:transparent !important; font-weight:500 !important;
      }
      section[data-testid="stSidebar"] button[kind="tertiary"]:hover {
          background:rgba(128,128,128,0.16) !important;
      }
      /* item activo */
      section[data-testid="stSidebar"] button[kind="secondary"] {
          background:rgba(128,128,128,0.20) !important; font-weight:700 !important;
      }
      /* espaciado compacto y uniforme entre items */
      section[data-testid="stSidebar"] div.stButton { margin-bottom:2px !important; }

      /* Reproductor de video más compacto (menos scroll para ver los insights) */
      [data-testid="stVideo"] video, .stVideo video {
          max-height: 300px !important; width: auto !important;
          max-width: 100% !important; margin: 0 auto; display: block; border-radius: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_BADGE = {
    "indexed":  '<span class="badge b-indexed">✓ indexado</span>',
    "indexing": '<span class="badge b-indexing">⏳ indexando…</span>',
    "uploaded": '<span class="badge b-uploaded">↥ subido</span>',
    "error":    '<span class="badge b-error">⚠ error</span>',
}


def fmt_ts(seconds) -> str:
    s = int(seconds or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _video_bytes(volume_path: str) -> bytes:
    return storage.download_video(volume_path)


def go(view: str, video_id: str = None, seek: int = None):
    st.session_state.view = view
    if video_id is not None:
        st.session_state.selected_video = video_id
        st.session_state["mv_pills"] = video_id   # mantener el navegador en sync
    if seek is not None:
        st.session_state.seek = seek


# ------------------------- Estado -------------------------
st.session_state.setdefault("view", "Mis Videos")
st.session_state.setdefault("selected_video", None)
st.session_state.setdefault("seek", 0)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("collection_id", "general")

# Config de INFRA/global por-sesión (editable en Configuración) → config.RUNTIME
INFRA_KEYS = ["lb_project", "lb_branch", "lb_endpoint", "lb_database", "lb_host",
              "uploads_volume", "frames_volume"]
if "cfg" not in st.session_state:
    st.session_state.cfg = {k: config.DEFAULTS[k] for k in INFRA_KEYS}
config.RUNTIME.update(st.session_state.cfg)

db = get_db()
NAV = ["Mis Videos", "Subir", "Buscar", "Agente", "Colecciones", "Configuración"]
NAV_ICONS = {"Subir": "📤", "Mis Videos": "🎬", "Buscar": "🔍",
             "Agente": "💬", "Colecciones": "📚", "Configuración": "⚙️"}

# --- Colecciones: cargar y fijar la activa + su config ---
collections, coll_error = [], None
try:
    collections = db.list_collections()
except Exception as e:
    coll_error = e
coll_ids = [c["collection_id"] for c in collections]
if st.session_state.collection_id not in coll_ids and coll_ids:
    st.session_state.collection_id = coll_ids[0]
active_coll = next((c for c in collections
                    if c["collection_id"] == st.session_state.collection_id), None)
config.ACTIVE_COLLECTION = dict(active_coll["config"]) if (active_coll and active_coll.get("config")) else {}
CID = st.session_state.collection_id

# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.title("🎬 videoRAG")
    st.caption("Transcripción · OCR · Entidades · Búsqueda · Agente")

    if coll_error:
        st.warning(f"Sin conexión a Lakebase: {coll_error}")
    elif coll_ids:
        labels = {c["collection_id"]: f"{c['name']} · {c['n_videos']} videos" for c in collections}
        sel = st.selectbox("📚 Colección", coll_ids,
                           index=coll_ids.index(CID),
                           format_func=lambda cid: labels.get(cid, cid))
        if sel != CID:
            st.session_state.collection_id = sel
            st.session_state.selected_video = None
            st.rerun()
        try:
            cnt = db.counts(CID)
            a, b, cc = st.columns(3)
            a.metric("Videos", cnt["videos"])
            b.metric("Indexados", cnt["indexed"])
            cc.metric("Fragmentos", cnt["chunks"])
        except Exception as e:
            st.warning(f"Error: {e}")

    st.divider()
    for item in NAV:
        active = item == st.session_state.view
        if st.button(item, key=f"nav_{item}", icon=NAV_ICONS[item],
                     type=("secondary" if active else "tertiary"),
                     use_container_width=True):
            st.session_state.view = item
            st.rerun()

    with st.expander("Detalles técnicos"):
        st.markdown(
            f"- **Colección:** `{CID}`\n"
            f"- **Lakebase (infra):** `{config.cfg('lb_project')}`\n"
            f"- **Embeddings:** `{config.cfg('embedding_endpoint')}`\n"
            f"- **LLM:** `{config.cfg('llm_endpoint')}`\n"
            f"- **Whisper:** `{config.cfg('whisper_model')}` · frames c/{config.cfg('frame_interval')}s\n"
            f"- **Compute:** `{config.cfg('compute')}` · job `{config.indexer_job_id()}`\n"
            f"- **Modo:** {'Databricks App' if config.IS_DATABRICKS_APP else 'Local'}")


view = st.session_state.view

# ============================================================
# SUBIR
# ============================================================
if view == "Subir":
    st.header("📤 Subir e indexar un video")
    coll_name = active_coll["name"] if active_coll else CID
    st.write(f"Se indexará en la colección **{coll_name}** con su configuración "
             "(modelos, Whisper, compute). Cámbiala en el sidebar o en «Colecciones».")
    st.caption(f"Modelos: embeddings `{config.cfg('embedding_endpoint')}` · "
               f"LLM `{config.cfg('llm_endpoint')}` · Whisper `{config.cfg('whisper_model')}` "
               f"· compute `{config.cfg('compute')}`")
    up = st.file_uploader("Archivo de video",
                          type=["mp4", "mov", "m4v", "avi", "mkv", "webm"])
    if up is not None:
        st.video(up)
        st.caption(f"{up.name} · {up.size/1e6:.1f} MB")
        if st.button("🚀 Subir e indexar", type="primary"):
            try:
                vid = uuid.uuid4().hex[:16]
                with st.spinner("Subiendo al Volume…"):
                    path = storage.upload_video(vid, up.name, up.getvalue(), CID)
                    user = config.get_workspace_client().current_user.me().user_name
                    db.create_video(vid, up.name, path, user, CID)
                with st.spinner("Lanzando job de indexación…"):
                    run_id = storage.trigger_indexing(
                        vid, path,
                        whisper_model=config.cfg("whisper_model"),
                        frame_interval=int(config.cfg("frame_interval")),
                        language=config.cfg("language"))
                    db.set_job_run(vid, run_id)
                st.success(f"Video enviado a indexar (run #{run_id}). "
                           "Puedes seguir el progreso en «Mis Videos».")
                st.session_state.selected_video = vid
                if st.button("Ir a Mis Videos"):
                    go("Mis Videos", vid); st.rerun()
            except Exception as e:
                st.error(f"Error al subir/indexar: {e}")

# ============================================================
# MIS VIDEOS
# ============================================================
elif view == "Mis Videos":
    st.header(f"🎬 Mis Videos · {active_coll['name'] if active_coll else CID}")
    videos = db.list_videos(CID)
    if not videos:
        st.info("Esta colección aún no tiene videos. Ve a «Subir» para agregar el primero.")
    else:
        # --- Navegador horizontal (tipo álbum) + Actualizar ---
        nav_c, act_c = st.columns([6, 1])
        with act_c:
            if st.button("🔄 Actualizar", use_container_width=True,
                         help="Refresca el estado de los videos en indexación."):
                for v in videos:
                    if v["status"] == "indexing" and v.get("job_run_id"):
                        try:
                            s = storage.run_status(v["job_run_id"])
                            if s["result"] in ("FAILED", "TIMEDOUT", "CANCELED"):
                                db.set_status(v["video_id"], "error", f"Job {s['result']}")
                        except Exception:
                            pass
                st.rerun()

        ids = [v["video_id"] for v in videos]
        vmap = {v["video_id"]: v for v in videos}
        emoji = {"indexed": "✓", "indexing": "⏳", "uploaded": "↥", "error": "⚠"}
        labels = {i: f'{emoji.get(vmap[i]["status"], "")} {vmap[i]["file_name"][:40]}' for i in ids}

        cur = st.session_state.selected_video if st.session_state.selected_video in ids else ids[0]
        if st.session_state.get("mv_pills") not in ids:
            st.session_state["mv_pills"] = cur
        with nav_c:
            st.pills("Biblioteca", ids, format_func=lambda i: labels[i],
                     selection_mode="single", key="mv_pills",
                     label_visibility="collapsed")
        target = st.session_state.get("mv_pills") or cur
        if target != st.session_state.selected_video:
            st.session_state.selected_video = target
            st.session_state.seek = 0
            st.rerun()

        vid = target
        v = vmap.get(vid) or db.get_video(vid)

        # --- Detalle a todo el ancho ---
        st.subheader(v["file_name"])
        st.markdown(STATUS_BADGE.get(v["status"], v["status"]), unsafe_allow_html=True)

        pcol, icol = st.columns([2, 3])
        with pcol:
            try:
                data = _video_bytes(v["volume_path"])
                st.video(data, start_time=int(st.session_state.seek or 0))
            except Exception as e:
                st.warning(f"No se pudo cargar el video: {e}")
        with icol:
            if v["status"] == "indexed":
                m = st.columns(4)
                m[0].metric("Duración", fmt_ts(v.get("duration_s")))
                m[1].metric("Idioma", v.get("language") or "—")
                m[2].metric("Segmentos", v.get("n_segments") or 0)
                m[3].metric("Frames OCR", v.get("n_frames") or 0)
                st.markdown("**📝 Resumen**")
                st.write(v.get("description") or "—")
                topics = v.get("topics") or []
                if topics:
                    st.markdown("**🏷️ Temas**")
                    st.markdown(" ".join(f'<span class="topic">{t}</span>' for t in topics),
                                unsafe_allow_html=True)
                ents = db.get_entities(vid)
                if ents:
                    st.markdown("**🔎 Entidades**")
                    by_type = {}
                    for e in ents:
                        by_type.setdefault(e["entity_type"], []).append(
                            f"{e['entity_value']}"
                            + (f" ({e['mentions']})" if e["mentions"] > 1 else ""))
                    for typ, vals in by_type.items():
                        st.markdown(f"- **{typ}:** " + ", ".join(vals))
            elif v["status"] == "indexing":
                st.info("⏳ Indexando… usa «Actualizar».")
                if v.get("job_run_id"):
                    try:
                        s = storage.run_status(v["job_run_id"])
                        st.caption(f"Job: {s['life_cycle']} / {s['result']} — "
                                   f"[ver run]({s['run_page_url']})")
                    except Exception:
                        pass
            elif v["status"] == "error":
                st.error(f"Error de indexación: {v.get('error_msg')}")

        if v["status"] == "indexed":
            t2, t3, t4 = st.tabs(["📑 Capítulos", "🗣️ Transcripción", "🔤 OCR"])
            with t2:
                for ch in db.get_chapters(vid):
                    cc = st.columns([1, 6])
                    if cc[0].button(f"▶ {fmt_ts(ch['start_time'])}",
                                    key=f"ch_{vid}_{ch['chapter_idx']}"):
                        go("Mis Videos", vid, int(ch["start_time"])); st.rerun()
                    cc[1].markdown(f"**{ch['title']}** — {ch['summary']}")
            with t3:
                for seg in db.get_transcript(vid):
                    cc = st.columns([1, 6])
                    if cc[0].button(f"▶ {fmt_ts(seg['start_time'])}",
                                    key=f"tr_{vid}_{seg['seq']}"):
                        go("Mis Videos", vid, int(seg["start_time"])); st.rerun()
                    cc[1].write(seg["text"])
            with t4:
                ocr = db.get_ocr(vid)
                if not ocr:
                    st.caption("Sin texto OCR detectado en los frames.")
                for i, o in enumerate(ocr):
                    cc = st.columns([1, 6])
                    if cc[0].button(f"▶ {fmt_ts(o['start_time'])}",
                                    key=f"ocr_{vid}_{i}"):
                        go("Mis Videos", vid, int(o["start_time"])); st.rerun()
                    cc[1].write(o["text"])

        st.divider()
        bcol1, bcol2 = st.columns(2)
        if bcol1.button("🔄 Reindexar", key=f"re_{vid}",
                        help="Reprocesa el video con la configuración actual de la colección."):
            try:
                run_id = storage.trigger_indexing(
                    vid, v["volume_path"],
                    whisper_model=config.cfg("whisper_model"),
                    frame_interval=int(config.cfg("frame_interval")),
                    language=config.cfg("language"))
                db.set_job_run(vid, run_id)
                st.success(f"Reindexando (run #{run_id})… sigue el estado con «Actualizar».")
                st.rerun()
            except Exception as e:
                st.error(f"Error al reindexar: {e}")
        if bcol2.button("🗑️ Eliminar video", key=f"del_{vid}"):
            storage.delete_video_file(v["volume_path"])
            db.delete_video(vid)
            st.session_state.selected_video = None
            st.session_state.pop("mv_pills", None)
            st.rerun()

# ============================================================
# BUSCAR
# ============================================================
elif view == "Buscar":
    st.header("🔍 Búsqueda semántica")
    q = st.text_input("¿Qué buscas en tus videos?",
                      placeholder="p. ej. dónde hablan de precios o promociones")
    c1, c2 = st.columns([3, 1])
    src = c1.radio("Fuente", ["Todo", "Transcripción", "OCR (texto en pantalla)"],
                   horizontal=True)
    top_k = c2.slider("Resultados", 5, 20, min(20, max(5, int(config.cfg("top_k")))))
    all_coll = st.checkbox("Buscar en todas las colecciones", value=False,
                           help="Ojo: colecciones con distinto modelo de embeddings pueden dar resultados mezclados.")
    src_map = {"Todo": None, "Transcripción": "transcript", "OCR (texto en pantalla)": "ocr"}
    if q:
        with st.spinner("Buscando…"):
            import fm
            emb = fm.embed(q)
            results = db.search(emb, top_k=top_k, source_type=src_map[src],
                                collection_id=None if all_coll else CID)
        st.caption(f"{len(results)} resultados")
        for i, r in enumerate(results):
            kind = "🗣️ transcripción" if r["source_type"] == "transcript" else "🔤 OCR"
            st.markdown(
                f'<div class="card"><span class="score">{r["score"]:.2f}</span> '
                f'<b>{r["file_name"]}</b> '
                f'<span class="meta">· {kind} · {fmt_ts(r["start_time"])}</span>'
                f'<div class="snippet">{r["text"][:280]}</div></div>',
                unsafe_allow_html=True)
            if st.button(f"▶ Ir a {fmt_ts(r['start_time'])} en el video", key=f"sr_{i}"):
                go("Mis Videos", r["video_id"], int(r["start_time"])); st.rerun()

# ============================================================
# AGENTE
# ============================================================
elif view == "Agente":
    st.header(f"💬 Agente · {active_coll['name'] if active_coll else CID}")
    st.caption("El agente responde con base en el contenido indexado de la colección "
               "activa y cita el video y minuto de cada dato.")
    ctop = st.columns([2, 1, 1])
    top_k = ctop[1].slider("Contexto (top-k)", 4, 16, min(16, max(4, int(config.cfg("top_k")))))
    agent_all = ctop[2].checkbox("Todas las colecciones", value=False)
    if ctop[0].button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for i, s in enumerate(m.get("sources", [])):
                kind = "transcripción" if s["source_type"] == "transcript" else "OCR"
                cc = st.columns([1, 6])
                if cc[0].button(f"▶ {fmt_ts(s['start_time'])}",
                                key=f"src_{id(m)}_{i}"):
                    go("Mis Videos", s["video_id"], int(s["start_time"])); st.rerun()
                cc[1].caption(f"[{i+1}] {s['file_name']} · {kind}")

    prompt = st.chat_input("Escribe tu pregunta…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Buscando en los videos…"):
                try:
                    hist = [{"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]]
                    ans, chunks = agent.answer(prompt, history=hist, top_k=top_k,
                                               collection_id=None if agent_all else CID)
                except Exception as e:
                    ans, chunks = f"⚠️ Error: {e}", []
            st.markdown(ans)
            for i, s in enumerate(chunks):
                kind = "transcripción" if s["source_type"] == "transcript" else "OCR"
                cc = st.columns([1, 6])
                if cc[0].button(f"▶ {fmt_ts(s['start_time'])}", key=f"newsrc_{i}"):
                    go("Mis Videos", s["video_id"], int(s["start_time"])); st.rerun()
                cc[1].caption(f"[{i+1}] {s['file_name']} · {kind}")
        st.session_state.messages.append(
            {"role": "assistant", "content": ans, "sources": chunks})

# ============================================================
# CONFIGURACIÓN
# ============================================================
elif view == "Colecciones":
    st.header("📚 Colecciones")
    st.caption("Cada colección agrupa videos y tiene su propia configuración de modelos, "
               "indexación y recuperación (persistida en la base de datos).")

    def _idx(options, val, fb=0):
        return options.index(val) if val in options else fb

    with st.expander("➕ Nueva colección"):
        nc1, nc2 = st.columns(2)
        new_name = nc1.text_input("Nombre", key="new_coll_name")
        new_desc = nc2.text_input("Descripción", key="new_coll_desc")
        if st.button("Crear colección", type="primary"):
            if not new_name.strip():
                st.warning("Ponle un nombre.")
            else:
                import re
                slug = re.sub(r"[^a-z0-9]+", "-", new_name.lower()).strip("-")[:32] or "col"
                cid_new = f"{slug}-{uuid.uuid4().hex[:6]}"
                try:
                    user = config.get_workspace_client().current_user.me().user_name
                    db.create_collection(cid_new, new_name.strip(), new_desc.strip(),
                                         config.default_collection_config(), user)
                    st.session_state.collection_id = cid_new
                    st.success(f"Colección «{new_name}» creada.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.subheader("Existentes")
    for col in collections:
        cc = {**config.default_collection_config(), **(col.get("config") or {})}
        mark = "✓ " if col["collection_id"] == CID else ""
        st.markdown(
            f'<div class="card"><b>{mark}{col["name"]}</b> '
            f'<span class="meta">· {col["n_videos"]} videos · <code>{col["collection_id"]}</code></span>'
            f'<div class="snippet">{col.get("description") or ""}<br>'
            f'emb <code>{cc["embedding_endpoint"]}</code> · llm <code>{cc["llm_endpoint"]}</code> · '
            f'whisper <code>{cc["whisper_model"]}</code> · compute <code>{cc["compute"]}</code></div></div>',
            unsafe_allow_html=True)

    st.divider()
    if active_coll:
        st.subheader(f"Editar «{active_coll['name']}» (colección activa)")
        cc = {**config.default_collection_config(), **(active_coll.get("config") or {})}
        with st.form("edit_coll"):
            e1, e2 = st.columns(2)
            ed_name = e1.text_input("Nombre", active_coll["name"])
            ed_desc = e2.text_input("Descripción", active_coll.get("description") or "")

            st.markdown("**🧠 Modelos**")
            m1, m2 = st.columns(2)
            emb_opts = config.EMBEDDING_OPTIONS + (
                [cc["embedding_endpoint"]] if cc["embedding_endpoint"] not in config.EMBEDDING_OPTIONS else [])
            cc["embedding_endpoint"] = m1.selectbox(
                "Embeddings (1024 dims)", emb_opts, index=_idx(emb_opts, cc["embedding_endpoint"]),
                help="Si lo cambias con videos ya indexados, reindexa los videos para mantener "
                     "consistencia de la búsqueda.")
            llm_opts = config.LLM_OPTIONS + (
                [cc["llm_endpoint"]] if cc["llm_endpoint"] not in config.LLM_OPTIONS else [])
            cc["llm_endpoint"] = m2.selectbox("LLM (agente + enriquecimiento)", llm_opts,
                                              index=_idx(llm_opts, cc["llm_endpoint"]))

            st.markdown("**🎬 Indexación**")
            cc["compute"] = st.radio(
                "Compute", config.COMPUTE_OPTIONS,
                index=_idx(config.COMPUTE_OPTIONS, str(cc["compute"]).upper()), horizontal=True,
                help="CPU (m5d.xlarge, faster-whisper int8): económico, más lento. "
                     "GPU (g4dn.xlarge T4, openai-whisper): mucho más rápido, mayor costo.")
            i1, i2, i3 = st.columns(3)
            cc["whisper_model"] = i1.selectbox("Whisper", config.WHISPER_OPTIONS,
                                               index=_idx(config.WHISPER_OPTIONS, cc["whisper_model"], 2))
            cc["frame_interval"] = i2.slider("Seg. entre frames (OCR)", 1, 15, int(cc["frame_interval"]))
            cc["language"] = i3.selectbox("Idioma", config.LANGUAGE_OPTIONS,
                                          index=_idx(config.LANGUAGE_OPTIONS, cc["language"]))

            st.markdown("**🔎 Recuperación**")
            cc["top_k"] = st.slider("top-k por defecto", 4, 20, int(cc["top_k"]))

            saved = st.form_submit_button("💾 Guardar configuración", type="primary")
        if saved:
            try:
                db.update_collection(CID, ed_name.strip() or active_coll["name"],
                                     ed_desc.strip(), cc)
                st.success("Colección actualizada.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        # Reindexado masivo (aplica la config actual a todos los videos de la colección)
        n_vids = active_coll["n_videos"]
        if n_vids > 0 and st.button(f"🔄 Reindexar los {n_vids} videos de esta colección",
                                    help="Reprocesa todos los videos con la config actual "
                                         "(úsalo tras cambiar el modelo de embeddings)."):
            ok, err = 0, 0
            for vv in db.list_videos(CID):
                try:
                    rid = storage.trigger_indexing(
                        vv["video_id"], vv["volume_path"],
                        whisper_model=config.cfg("whisper_model"),
                        frame_interval=int(config.cfg("frame_interval")),
                        language=config.cfg("language"))
                    db.set_job_run(vv["video_id"], rid); ok += 1
                except Exception:
                    err += 1
            st.success(f"Relanzados {ok} videos a reindexar" + (f" · {err} con error" if err else "") + ".")

        if CID == "general":
            st.caption("La colección «General» no se puede borrar.")
        elif active_coll["n_videos"] > 0:
            st.caption("Para borrar esta colección, primero elimina sus videos en «Mis Videos».")
        elif st.button("🗑️ Borrar colección"):
            db.delete_collection(CID)
            st.session_state.collection_id = "general"
            st.rerun()

# ============================================================
# CONFIGURACIÓN (infraestructura)
# ============================================================
elif view == "Configuración":
    st.header("⚙️ Configuración · infraestructura")
    st.caption("Ajustes globales de infraestructura para esta sesión (Lakebase y volúmenes). "
               "La configuración de modelos/indexación vive en cada colección.")
    c = dict(st.session_state.cfg)

    with st.form("cfg_form"):
        st.subheader("📁 Almacenamiento (UC Volumes)")
        st.caption("Volumen base; cada colección guarda sus videos en una subcarpeta propia. "
                   "El service principal necesita `WRITE VOLUME` en uploads y `READ VOLUME` en frames.")
        c["uploads_volume"] = st.text_input("Volumen de videos (uploads)", c["uploads_volume"])
        c["frames_volume"] = st.text_input("Volumen de frames (OCR)", c["frames_volume"])

        st.subheader("🗄️ Lakebase (avanzado)")
        st.caption("Apunta la app a otro proyecto Lakebase. El SP necesita rol Postgres + "
                   "`CAN USE` en ese proyecto, y el esquema debe existir (usa «Inicializar esquema»).")
        l1, l2 = st.columns(2)
        c["lb_project"] = l1.text_input("Proyecto", c["lb_project"])
        c["lb_database"] = l2.text_input("Base de datos", c["lb_database"])
        l3, l4 = st.columns(2)
        c["lb_branch"] = l3.text_input("Branch", c["lb_branch"])
        c["lb_endpoint"] = l4.text_input("Endpoint", c["lb_endpoint"])
        c["lb_host"] = st.text_input(
            "Host (opcional)", c.get("lb_host", ""),
            help="Host del endpoint Postgres. Vacío = proyecto por defecto. Requerido para otro proyecto.")
        applied = st.form_submit_button("💾 Aplicar", type="primary")

    if applied:
        lb_changed = any(c[k] != st.session_state.cfg[k]
                         for k in ("lb_project", "lb_branch", "lb_endpoint", "lb_database", "lb_host"))
        st.session_state.cfg = c
        config.RUNTIME.update(c)
        config._HOST_CACHE.clear()
        if lb_changed:
            db.reset()
        st.success("Configuración aplicada para esta sesión.")

    st.divider()
    st.subheader("🔌 Diagnóstico de Lakebase")
    d1, d2, d3 = st.columns(3)
    if d1.button("Probar conexión", use_container_width=True):
        try:
            db.reset()
            info = db.test_connection()
            st.success(f"Conectado a `{info['database']}` en `{info['host']}` como `{info['user']}`.")
            st.caption(info["version"])
            st.info("✓ Esquema presente." if info["schema_ready"]
                    else "El esquema no existe aquí. Usa «Inicializar esquema».")
        except Exception as e:
            st.error(f"No se pudo conectar: {e}")
    if d2.button("Inicializar esquema", use_container_width=True):
        try:
            db.ensure_schema()
            st.success("Esquema creado/verificado (pgvector + tablas + colección General).")
        except Exception as e:
            st.error(f"Error al crear el esquema: {e}")
    if d3.button("Restaurar infra por defecto", use_container_width=True):
        st.session_state.cfg = {k: config.DEFAULTS[k] for k in INFRA_KEYS}
        config.RUNTIME.update(st.session_state.cfg)
        config._HOST_CACHE.clear()
        db.reset()
        st.rerun()

    st.subheader("📁 Diagnóstico de almacenamiento")
    if st.button("Probar volumen de videos (escritura)", use_container_width=True):
        try:
            info = storage.test_volume(config.cfg("uploads_volume"))
            st.success(f"✓ Escritura OK en `{info['path']}`.")
        except Exception as e:
            st.error(f"No se pudo escribir en el volumen: {e}")

    with st.expander("Valores efectivos actuales"):
        st.json({**{k: config.cfg(k) for k in INFRA_KEYS},
                 **{k: config.cfg(k) for k in sorted(config.COLLECTION_KEYS)},
                 "coleccion_activa": CID,
                 "modo": "Databricks App" if config.IS_DATABRICKS_APP else "Local"})
