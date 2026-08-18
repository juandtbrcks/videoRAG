"""Agente RAG sobre videos con LangGraph + tool-calling y MLflow tracing.

A diferencia del RAG lineal (agent.py), aquí el LLM decide QUÉ herramientas usar y
CUÁNTAS veces: puede buscar semánticamente, listar videos, leer capítulos/entidades
de un video concreto o rastrear una entidad por toda la biblioteca, en un bucle
agente↔herramientas (grafo de estados). El LLM se invoca vía el endpoint OpenAI-
compatible de Databricks (sin `temperature`, que claude-sonnet-5 rechaza).

Mantiene la misma firma pública que agent.py: answer(...) -> (texto, chunks), donde
`chunks` son los fragmentos recuperados (para las citas clicables de la app).
"""
import json
import os
from typing import List, Dict, Optional, Tuple, TypedDict

import config
import fm
from db import get_db
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------------------------------
# MLflow tracing (best-effort; si falla, el agente sigue funcionando)
# --------------------------------------------------------------------------
_TRACING = False
try:
    import mlflow  # noqa
    _exp = os.environ.get("MLFLOW_EXPERIMENT")
    if _exp:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(_exp)
        mlflow.openai.autolog()   # traza cada llamada al LLM/embeddings
        _TRACING = True
except Exception:
    _TRACING = False


def _trace(name):
    def deco(fn):
        if _TRACING:
            try:
                return mlflow.trace(name=name)(fn)
            except Exception:
                return fn
        return fn
    return deco


def _fmt_ts(seconds) -> str:
    s = int(seconds or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------------
# Definición de herramientas expuestas al LLM (formato OpenAI tools)
# --------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "buscar_en_videos",
        "description": ("Búsqueda semántica en el contenido indexado (transcripción de audio y "
                        "texto OCR en pantalla) de los videos. Úsala SIEMPRE para fundamentar "
                        "respuestas sobre lo que se dice o aparece en los videos. Puedes acotar a "
                        "un video con video_id."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Consulta en lenguaje natural."},
            "video_id": {"type": "string", "description": "Opcional: limitar a un video."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "listar_videos",
        "description": ("Lista los videos indexados (video_id, nombre, duración, idioma, temas). "
                        "Úsala para descubrir qué videos existen o para resolver un nombre de "
                        "video a su video_id antes de acotar otras herramientas."),
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "capitulos_de_video",
        "description": "Devuelve los capítulos (título, resumen y tiempos) de un video dado su video_id.",
        "parameters": {"type": "object", "properties": {
            "video_id": {"type": "string"}}, "required": ["video_id"]}}},
    {"type": "function", "function": {
        "name": "entidades_de_video",
        "description": ("Devuelve las entidades detectadas (personas, organizaciones, productos, "
                        "temas) de un video dado su video_id, con su número de menciones."),
        "parameters": {"type": "object", "properties": {
            "video_id": {"type": "string"}}, "required": ["video_id"]}}},
    {"type": "function", "function": {
        "name": "buscar_entidad",
        "description": ("Rastrea una entidad (nombre de persona, marca, tema…) por toda la "
                        "biblioteca y devuelve en qué videos aparece y cuántas veces."),
        "parameters": {"type": "object", "properties": {
            "termino": {"type": "string"}}, "required": ["termino"]}}},
]

SYSTEM = (
    "Eres un asistente experto en una biblioteca de videos ya indexados (transcripción de audio "
    "y texto OCR de los cuadros). Respondes SIEMPRE en español, claro y conciso.\n\n"
    "Tienes herramientas para buscar semánticamente, listar videos y leer capítulos/entidades. "
    "Fundamenta TODA afirmación en los resultados de las herramientas; si la información no está, "
    "dilo honestamente y no inventes. Antes de responder sobre el contenido, usa 'buscar_en_videos' "
    "(una o varias veces, refinando la consulta si hace falta). Para preguntas sobre un video "
    "concreto por nombre, primero usa 'listar_videos' para obtener su video_id.\n\n"
    "Cita cada dato con el número entre corchetes que aparece en los resultados de búsqueda, "
    "p.ej. [1], e indica el video y el minuto (mm:ss) cuando sea útil."
)


class _State(TypedDict):
    messages: list


# --------------------------------------------------------------------------
def answer(query: str, history: Optional[List[Dict]] = None,
           top_k: int = 8, video_id: Optional[str] = None,
           collection_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
    """Ejecuta el agente LangGraph y devuelve (texto, chunks_recuperados)."""
    db = get_db()
    retrieved: List[Dict] = []  # acumula chunks para las citas de la app

    # ---- herramientas (closures sobre colección/top_k/retrieved) ----
    @_trace("buscar_en_videos")
    def buscar_en_videos(query: str, video_id: Optional[str] = None) -> str:
        emb = fm.embed(query)
        rows = db.search(emb, top_k=top_k, video_id=video_id, collection_id=collection_id)
        if not rows:
            return "Sin resultados para esa búsqueda."
        blocks = []
        for c in rows:
            idx = len(retrieved) + 1
            retrieved.append(c)
            kind = "transcripción" if c["source_type"] == "transcript" else "texto en pantalla (OCR)"
            blocks.append(f"[{idx}] Video: {c['file_name']} · {_fmt_ts(c['start_time'])} · "
                          f"{kind} (score {c['score']:.2f})\n{c['text']}")
        return "\n\n".join(blocks)

    @_trace("listar_videos")
    def listar_videos() -> str:
        vids = db.list_videos(collection_id=collection_id)
        vids = [v for v in vids if v.get("status") == "indexed"] or vids
        if not vids:
            return "No hay videos indexados en el alcance actual."
        out = []
        for v in vids[:40]:
            temas = ", ".join(v.get("topics") or []) if v.get("topics") else ""
            dur = _fmt_ts(v.get("duration_s")) if v.get("duration_s") else "?"
            out.append(f"- video_id={v['video_id']} · {v['file_name']} · dur {dur} · "
                       f"idioma {v.get('language') or '?'}" + (f" · temas: {temas}" if temas else ""))
        return "\n".join(out)

    @_trace("capitulos_de_video")
    def capitulos_de_video(video_id: str) -> str:
        caps = db.get_chapters(video_id)
        if not caps:
            return "Ese video no tiene capítulos registrados."
        return "\n".join(f"[{c['chapter_idx']}] {_fmt_ts(c['start_time'])}–{_fmt_ts(c['end_time'])} "
                         f"{c['title']}: {c['summary']}" for c in caps)

    @_trace("entidades_de_video")
    def entidades_de_video(video_id: str) -> str:
        ents = db.get_entities(video_id)
        if not ents:
            return "Ese video no tiene entidades registradas."
        return "\n".join(f"- {e['entity_type']}: {e['entity_value']} ({e['mentions']} menciones)"
                         for e in ents[:50])

    @_trace("buscar_entidad")
    def buscar_entidad(termino: str) -> str:
        rows = db.search_entities(termino, collection_id=collection_id)
        if not rows:
            return f"No encontré la entidad '{termino}' en la biblioteca."
        return "\n".join(f"- '{r['entity_value']}' ({r['entity_type']}) en {r['file_name']} "
                         f"[video_id={r['video_id']}] · {r['mentions']} menciones" for r in rows)

    toolkit = {
        "buscar_en_videos": buscar_en_videos, "listar_videos": listar_videos,
        "capitulos_de_video": capitulos_de_video, "entidades_de_video": entidades_de_video,
        "buscar_entidad": buscar_entidad,
    }

    client = fm._client()
    model = config.cfg("llm_endpoint")

    # ---- nodos del grafo ----
    def agent_node(state: _State) -> _State:
        resp = client.chat.completions.create(
            model=model, messages=state["messages"],
            tools=TOOLS_SCHEMA, tool_choice="auto", max_tokens=1500)
        m = resp.choices[0].message
        text = fm._content_text(m.content)
        # espejo del contrato del endpoint: content=null cuando solo hay tool_calls
        msg: Dict = {"role": "assistant", "content": text if text else None}
        if getattr(m, "tool_calls", None):
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in m.tool_calls]
        return {"messages": state["messages"] + [msg]}

    def tools_node(state: _State) -> _State:
        last = state["messages"][-1]
        new = []
        for tc in last.get("tool_calls", []):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            fn = toolkit.get(name)
            try:
                result = fn(**args) if fn else f"Herramienta desconocida: {name}"
            except Exception as e:
                result = f"Error ejecutando {name}: {e}"
            new.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)[:6000]})
        return {"messages": state["messages"] + new}

    def route(state: _State):
        return "tools" if state["messages"][-1].get("tool_calls") else END

    g = StateGraph(_State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    graph = g.compile()

    # ---- mensajes iniciales ----
    messages: List[Dict] = [{"role": "system", "content": SYSTEM}]
    if history:
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
    scope = f" (limita la búsqueda al video_id={video_id})" if video_id else ""
    messages.append({"role": "user", "content": query + scope})

    run = _trace("agente_video_rag")(lambda: graph.invoke(
        {"messages": messages}, {"recursion_limit": 12}))
    try:
        final = run()
        text = final["messages"][-1].get("content") or ""
        if not text.strip():
            text = "No pude generar una respuesta. Intenta reformular la pregunta."
    except Exception as e:
        # fallback duro: recuperación simple + una llamada al LLM
        try:
            ctx = buscar_en_videos(query, video_id=video_id)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"CONTEXTO:\n{ctx}\n\n---\nPREGUNTA: {query}"}]
            text = fm.chat(msgs, max_tokens=1024)
        except Exception:
            text = f"⚠️ Error del agente: {e}"

    # dedup de chunks preservando orden (para las citas)
    seen, chunks = set(), []
    for c in retrieved:
        key = (c["video_id"], c["source_type"], round(float(c["start_time"] or 0), 1))
        if key not in seen:
            seen.add(key)
            chunks.append(c)
    return text, chunks
