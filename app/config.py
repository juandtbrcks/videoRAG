"""Configuración y autenticación dual-mode para Video Indexer (Databricks App).

- Local: usa un perfil del Databricks CLI (env DATABRICKS_PROFILE).
- Databricks App: usa el service principal inyectado por el runtime.

Los valores por defecto vienen de env (app.yaml). La pestaña «Configuración» de la
app puede sobreescribirlos por-sesión vía el dict RUNTIME (config.cfg(...)).

Nota: el proyecto Lakebase `video-indexer` es de tipo AUTOSCALING, que NO soporta
el binding de recurso `database` de la App. Por eso PGUSER es el client_id del SP
(su rol de Postgres) y el password de Postgres es el token OAuth del SP.
"""
import os
from functools import lru_cache

from databricks.sdk import WorkspaceClient

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))

# --- Unity Catalog storage (defaults) --------------------------------------
UC_CATALOG = os.environ.get("UC_CATALOG", "jgworkspaceclassic_catalog")
UC_SCHEMA = os.environ.get("UC_SCHEMA", "video_indexer")
_DEFAULT_UPLOADS = os.environ.get("UPLOADS_VOLUME", f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/uploads")
_DEFAULT_FRAMES = os.environ.get("FRAMES_VOLUME", f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/frames")

# --- Valores por defecto (env-overridable). RUNTIME los puede sobreescribir. ---
DEFAULTS = {
    "embedding_endpoint": os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en"),
    "llm_endpoint":       os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-5"),
    "lb_project":         os.environ.get("LB_PROJECT", "video-indexer"),
    "lb_branch":          os.environ.get("LB_BRANCH", "production"),
    "lb_endpoint":        os.environ.get("LB_ENDPOINT", "primary"),
    "lb_database":        os.environ.get("PGDATABASE", "databricks_postgres"),
    "lb_host":            os.environ.get("PGHOST", ""),
    "uploads_volume":     _DEFAULT_UPLOADS,
    "frames_volume":      _DEFAULT_FRAMES,
    "compute":            os.environ.get("INDEX_COMPUTE", "CPU"),
    "whisper_model":      os.environ.get("WHISPER_MODEL", "small"),
    "frame_interval":     int(os.environ.get("FRAME_INTERVAL", "5")),
    "language":           os.environ.get("LANGUAGE", "auto"),
    "top_k":              int(os.environ.get("TOP_K", "8")),
}

# Config a nivel COLECCIÓN (se guarda por colección en la BD) vs INFRA/global.
COLLECTION_KEYS = {"embedding_endpoint", "llm_endpoint", "whisper_model",
                   "frame_interval", "language", "compute", "top_k"}

# RUNTIME  = overrides de infra/global por-sesión (lb_*, volúmenes)
# ACTIVE_COLLECTION = config de la colección activa (poblada por app.py)
RUNTIME: dict = {}
ACTIVE_COLLECTION: dict = {}


def cfg(key):
    """Valor efectivo. Claves de colección: ACTIVE_COLLECTION > DEFAULTS.
    Claves de infra/global: RUNTIME > DEFAULTS."""
    if key in COLLECTION_KEYS:
        v = ACTIVE_COLLECTION.get(key)
        return v if v not in (None, "") else DEFAULTS.get(key)
    v = RUNTIME.get(key)
    return v if v not in (None, "") else DEFAULTS.get(key)


def default_collection_config() -> dict:
    """Config por defecto para una colección nueva."""
    return {k: DEFAULTS[k] for k in COLLECTION_KEYS}


# --- Opciones para los selectores de la UI --------------------------------
# Todos los modelos de embeddings deben ser de 1024 dims (coincide con el
# esquema vector(1024)). gte-large-en y bge-large-en son 1024.
EMBEDDING_OPTIONS = ["databricks-gte-large-en", "databricks-bge-large-en"]
EMBEDDING_DIM = 1024
LLM_OPTIONS = [
    "databricks-claude-sonnet-5",
    "databricks-claude-sonnet-4-6",
    "databricks-claude-opus-5",
    "databricks-claude-haiku-4-5",
    "databricks-gpt-5",
    "databricks-meta-llama-3-3-70b-instruct",
]
WHISPER_OPTIONS = ["tiny", "base", "small", "medium"]
LANGUAGE_OPTIONS = ["auto", "es", "en", "pt", "fr"]
COMPUTE_OPTIONS = ["CPU", "GPU"]

# --- Indexing Jobs (CPU y GPU) --------------------------------------------
INDEXER_JOB_ID_CPU = int(os.environ.get("INDEXER_JOB_ID", "552091141704054"))
INDEXER_JOB_ID_GPU = int(os.environ.get("INDEXER_JOB_ID_GPU", "1108962426104160"))


def indexer_job_id() -> int:
    """job_id según el compute seleccionado (CPU/GPU)."""
    return INDEXER_JOB_ID_GPU if str(cfg("compute")).upper() == "GPU" else INDEXER_JOB_ID_CPU


@lru_cache(maxsize=1)
def get_workspace_client() -> WorkspaceClient:
    if IS_DATABRICKS_APP:
        return WorkspaceClient()
    profile = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
    return WorkspaceClient(profile=profile)


def get_oauth_token() -> str:
    """Token OAuth para Lakebase y serving endpoints."""
    client = get_workspace_client()
    auth = client.config.authenticate()
    if auth and "Authorization" in auth:
        return auth["Authorization"].replace("Bearer ", "")
    return client.config.token


def get_workspace_host() -> str:
    if IS_DATABRICKS_APP:
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        return host
    return get_workspace_client().config.host


def lb_endpoint_name() -> str:
    return f"projects/{cfg('lb_project')}/branches/{cfg('lb_branch')}/endpoints/{cfg('lb_endpoint')}"


_HOST_CACHE: dict = {}


def resolve_pg_host() -> str:
    """Resuelve el host del endpoint Lakebase para el proyecto configurado.

    Prioridad:
      1. Host explícito en la config (campo «Host» / env PGHOST).
      2. Proyecto por defecto → PGHOST inyectado por la App (sin llamar a la API).
      3. Resolver vía el SDK de Databricks (requiere soporte `postgres`).
    """
    explicit = cfg("lb_host")
    if explicit:
        return explicit
    if cfg("lb_project") == DEFAULTS["lb_project"] and os.environ.get("PGHOST"):
        return os.environ["PGHOST"]
    coords = lb_endpoint_name()
    if coords in _HOST_CACHE:
        return _HOST_CACHE[coords]
    w = get_workspace_client()
    if not hasattr(w, "postgres"):
        raise RuntimeError(
            "El SDK de Databricks instalado no puede resolver el host de Lakebase "
            "automáticamente. Indica el host manualmente en Configuración → Lakebase "
            "(campo «Host»).")
    ep = w.postgres.get_endpoint(name=coords)
    host = ep.status.hosts.host
    _HOST_CACHE[coords] = host
    return host


def get_pg_connection_params() -> dict:
    host = resolve_pg_host()
    if IS_DATABRICKS_APP:
        user = os.environ["PGUSER"]
    else:
        user = os.environ.get("PGUSER") or get_workspace_client().current_user.me().user_name
    return {
        "host": host,
        "dbname": cfg("lb_database"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "user": user,
        "sslmode": "require",
        "application_name": "video-indexer-app",
    }
