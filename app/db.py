"""Capa de acceso a Lakebase (pgvector) para Video Indexer."""
import time
from typing import List, Dict, Optional

import psycopg2
import psycopg2.extras

import config

_TOKEN_TTL_SECONDS = 45 * 60


class Database:
    def __init__(self):
        self._conn = None
        self._created_at = 0.0

    # ------------------------- Conexión -------------------------
    def _get_conn(self):
        age = time.time() - self._created_at
        if self._conn is not None and age < _TOKEN_TTL_SECONDS:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self._conn
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        params = config.get_pg_connection_params()
        token = config.get_oauth_token()
        self._conn = psycopg2.connect(password=token, **params)
        self._conn.set_session(autocommit=True)
        self._created_at = time.time()
        return self._conn

    def reset(self):
        """Cierra la conexión cacheada (p.ej. al cambiar de proyecto Lakebase)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._created_at = 0.0

    def test_connection(self) -> Dict:
        """Prueba la conexión y reporta host/usuario/versión + si el esquema existe."""
        params = config.get_pg_connection_params()
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
            cur.execute("SELECT to_regclass('public.videos') IS NOT NULL, "
                        "to_regclass('public.video_chunks') IS NOT NULL")
            has_videos, has_chunks = cur.fetchone()
        return {"host": params["host"], "user": params["user"],
                "database": params["dbname"], "version": ver,
                "schema_ready": bool(has_videos and has_chunks)}

    def ensure_schema(self):
        """Crea la extensión pgvector y las tablas si no existen (idempotente)."""
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS collections (
            collection_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
            config JSONB NOT NULL DEFAULT '{{}}'::jsonb, created_by TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW());
        INSERT INTO collections (collection_id, name, description, created_by)
            VALUES ('general', 'General', 'Colección por defecto', 'system')
            ON CONFLICT (collection_id) DO NOTHING;
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, volume_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded', duration_s DOUBLE PRECISION,
            language TEXT, description TEXT, topics TEXT[], n_segments INT DEFAULT 0,
            n_frames INT DEFAULT 0, error_msg TEXT, job_run_id BIGINT, uploaded_by TEXT,
            collection_id TEXT, uploaded_at TIMESTAMPTZ DEFAULT NOW(), indexed_at TIMESTAMPTZ);
        CREATE INDEX IF NOT EXISTS idx_videos_collection ON videos(collection_id);
        CREATE TABLE IF NOT EXISTS video_chunks (
            id BIGSERIAL PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
            source_type TEXT NOT NULL, seq INT, start_time DOUBLE PRECISION,
            end_time DOUBLE PRECISION, text TEXT NOT NULL,
            embedding vector({config.EMBEDDING_DIM}), created_at TIMESTAMPTZ DEFAULT NOW());
        CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON video_chunks
            USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
        CREATE INDEX IF NOT EXISTS idx_chunks_video ON video_chunks(video_id);
        CREATE TABLE IF NOT EXISTS video_entities (
            id BIGSERIAL PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
            entity_type TEXT, entity_value TEXT, mentions INT DEFAULT 1);
        CREATE INDEX IF NOT EXISTS idx_entities_video ON video_entities(video_id);
        CREATE TABLE IF NOT EXISTS video_chapters (
            id BIGSERIAL PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
            chapter_idx INT, start_time DOUBLE PRECISION, end_time DOUBLE PRECISION,
            title TEXT, summary TEXT);
        CREATE INDEX IF NOT EXISTS idx_chapters_video ON video_chapters(video_id);
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            for stmt in ddl.split(";"):
                if stmt.strip():
                    cur.execute(stmt)

    def _query(self, sql: str, args=None) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, args or ())
            return [dict(r) for r in cur.fetchall()]

    def _exec(self, sql: str, args=None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, args or ())

    # ------------------------- Colecciones -------------------------
    def list_collections(self) -> List[Dict]:
        return self._query(
            "SELECT c.collection_id, c.name, c.description, c.config, c.created_at, "
            "  (SELECT COUNT(*) FROM videos v WHERE v.collection_id = c.collection_id) AS n_videos "
            "FROM collections c ORDER BY c.created_at")

    def get_collection(self, collection_id: str) -> Optional[Dict]:
        rows = self._query("SELECT * FROM collections WHERE collection_id=%s", (collection_id,))
        return rows[0] if rows else None

    def create_collection(self, collection_id: str, name: str, description: str,
                          config: dict, created_by: str):
        import json
        self._exec(
            "INSERT INTO collections (collection_id, name, description, config, created_by) "
            "VALUES (%s,%s,%s,%s::jsonb,%s) ON CONFLICT (collection_id) DO NOTHING",
            (collection_id, name, description, json.dumps(config), created_by))

    def update_collection(self, collection_id: str, name: str, description: str, config: dict):
        import json
        self._exec(
            "UPDATE collections SET name=%s, description=%s, config=%s::jsonb WHERE collection_id=%s",
            (name, description, json.dumps(config), collection_id))

    def delete_collection(self, collection_id: str):
        self._exec("DELETE FROM collections WHERE collection_id=%s", (collection_id,))

    # ------------------------- Videos -------------------------
    def create_video(self, video_id: str, file_name: str, volume_path: str,
                     uploaded_by: str, collection_id: str):
        self._exec(
            "INSERT INTO videos (video_id, file_name, volume_path, status, uploaded_by, collection_id) "
            "VALUES (%s,%s,%s,'uploaded',%s,%s) ON CONFLICT (video_id) DO NOTHING",
            (video_id, file_name, volume_path, uploaded_by, collection_id),
        )

    def set_job_run(self, video_id: str, run_id: int):
        self._exec("UPDATE videos SET job_run_id=%s, status='indexing' WHERE video_id=%s",
                   (run_id, video_id))

    def set_status(self, video_id: str, status: str, error_msg: Optional[str] = None):
        self._exec("UPDATE videos SET status=%s, error_msg=%s WHERE video_id=%s",
                   (status, error_msg, video_id))

    def list_videos(self, collection_id: Optional[str] = None) -> List[Dict]:
        where = "WHERE collection_id = %s" if collection_id else ""
        args = (collection_id,) if collection_id else ()
        return self._query(
            "SELECT video_id, file_name, volume_path, status, duration_s, language, "
            "description, topics, n_segments, n_frames, error_msg, job_run_id, "
            "uploaded_by, collection_id, uploaded_at, indexed_at "
            f"FROM videos {where} ORDER BY uploaded_at DESC", args
        )

    def get_video(self, video_id: str) -> Optional[Dict]:
        rows = self._query("SELECT * FROM videos WHERE video_id=%s", (video_id,))
        return rows[0] if rows else None

    def delete_video(self, video_id: str):
        self._exec("DELETE FROM videos WHERE video_id=%s", (video_id,))

    def counts(self, collection_id: Optional[str] = None) -> Dict:
        if collection_id:
            rows = self._query(
                "SELECT "
                "(SELECT COUNT(*) FROM videos WHERE collection_id=%s) AS videos, "
                "(SELECT COUNT(*) FROM videos WHERE collection_id=%s AND status='indexed') AS indexed, "
                "(SELECT COUNT(*) FROM video_chunks ch JOIN videos v ON v.video_id=ch.video_id "
                "  WHERE v.collection_id=%s) AS chunks",
                (collection_id, collection_id, collection_id))
        else:
            rows = self._query(
                "SELECT (SELECT COUNT(*) FROM videos) AS videos, "
                "(SELECT COUNT(*) FROM videos WHERE status='indexed') AS indexed, "
                "(SELECT COUNT(*) FROM video_chunks) AS chunks")
        return rows[0] if rows else {"videos": 0, "indexed": 0, "chunks": 0}

    # ------------------------- Insights -------------------------
    def get_chapters(self, video_id: str) -> List[Dict]:
        return self._query(
            "SELECT chapter_idx, start_time, end_time, title, summary "
            "FROM video_chapters WHERE video_id=%s ORDER BY chapter_idx", (video_id,))

    def get_entities(self, video_id: str) -> List[Dict]:
        return self._query(
            "SELECT entity_type, entity_value, mentions FROM video_entities "
            "WHERE video_id=%s ORDER BY mentions DESC, entity_value", (video_id,))

    def get_transcript(self, video_id: str) -> List[Dict]:
        return self._query(
            "SELECT seq, start_time, end_time, text FROM video_chunks "
            "WHERE video_id=%s AND source_type='transcript' ORDER BY start_time", (video_id,))

    def get_ocr(self, video_id: str) -> List[Dict]:
        return self._query(
            "SELECT start_time, text FROM video_chunks "
            "WHERE video_id=%s AND source_type='ocr' ORDER BY start_time", (video_id,))

    # ------------------------- Búsqueda vectorial -------------------------
    def search(self, embedding: List[float], top_k: int = 8,
               video_id: Optional[str] = None, source_type: Optional[str] = None,
               collection_id: Optional[str] = None) -> List[Dict]:
        emb = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
        where = ["c.embedding IS NOT NULL"]
        # Placeholder order: score-emb, [video_id], [source_type], [collection], order-emb, limit
        args: list = [emb]
        if video_id:
            where.append("c.video_id = %s"); args.append(video_id)
        if source_type:
            where.append("c.source_type = %s"); args.append(source_type)
        if collection_id:
            where.append("v.collection_id = %s"); args.append(collection_id)
        args += [emb, top_k]
        sql = f"""
            SELECT c.video_id, c.source_type, c.start_time, c.end_time, c.text,
                   v.file_name,
                   1 - (c.embedding <=> %s::vector) AS score
            FROM video_chunks c
            JOIN videos v ON v.video_id = c.video_id
            WHERE {' AND '.join(where)}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        return self._query(sql, tuple(args))


_db = None


def get_db() -> "Database":
    global _db
    if _db is None:
        _db = Database()
    return _db
