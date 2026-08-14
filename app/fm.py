"""Cliente de Foundation Models (embeddings + LLM) vía API OpenAI-compatible."""
from typing import List, Dict

from openai import OpenAI

import config


def _client() -> OpenAI:
    host = config.get_workspace_host()
    token = config.get_oauth_token()
    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")


def embed(text: str) -> List[float]:
    resp = _client().embeddings.create(model=config.cfg("embedding_endpoint"), input=text)
    return resp.data[0].embedding


def _content_text(content) -> str:
    """Los modelos de razonamiento pueden devolver content como lista de bloques."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", "") or "")
            elif isinstance(b, str):
                parts.append(b)
            else:
                parts.append(getattr(b, "text", "") or "")
        return "".join(parts)
    return str(content or "")


def chat(messages: List[Dict], max_tokens: int = 1024) -> str:
    # nota: los modelos de razonamiento (claude-sonnet-5) no aceptan 'temperature'
    resp = _client().chat.completions.create(
        model=config.cfg("llm_endpoint"),
        messages=messages,
        max_tokens=max_tokens,
    )
    return _content_text(resp.choices[0].message.content)
