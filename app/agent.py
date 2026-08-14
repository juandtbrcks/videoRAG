"""Agente RAG sobre el contenido indexado de todos los videos (pgvector + LLM)."""
from typing import List, Dict, Optional, Tuple

import fm
from db import get_db


def _fmt_ts(seconds) -> str:
    s = int(seconds or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def retrieve(query: str, top_k: int = 8, video_id: Optional[str] = None,
             collection_id: Optional[str] = None) -> List[Dict]:
    emb = fm.embed(query)
    return get_db().search(emb, top_k=top_k, video_id=video_id, collection_id=collection_id)


def _build_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        kind = "transcripción" if c["source_type"] == "transcript" else "texto en pantalla (OCR)"
        blocks.append(
            f"[{i}] Video: {c['file_name']} · {_fmt_ts(c['start_time'])} · {kind}\n{c['text']}"
        )
    return "\n\n".join(blocks)


SYSTEM = (
    "Eres un asistente que responde preguntas sobre una biblioteca de videos ya indexados "
    "(transcripción de audio y texto OCR de los frames). Responde SIEMPRE en español, claro y conciso.\n\n"
    "Usa EXCLUSIVAMENTE la información del CONTEXTO. Si no está ahí, dilo honestamente y no inventes. "
    "Cita cada afirmación con su número entre corchetes, p.ej. [1], indicando el video y el minuto. "
    "Cuando sea útil, menciona el video y el timestamp (mm:ss) donde aparece la información."
)


def answer(query: str, history: Optional[List[Dict]] = None,
           top_k: int = 8, video_id: Optional[str] = None,
           collection_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
    chunks = retrieve(query, top_k=top_k, video_id=video_id, collection_id=collection_id)
    if not chunks:
        return ("No encontré contenido indexado relacionado con tu pregunta. "
                "¿Ya subiste e indexaste algún video sobre ese tema?", [])
    context = _build_context(chunks)
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": f"CONTEXTO:\n{context}\n\n---\nPREGUNTA: {query}"})
    text = fm.chat(messages, max_tokens=1024)
    return text, chunks
