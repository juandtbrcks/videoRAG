-- ============================================================
-- videoRAG — Lakebase (PostgreSQL + pgvector) schema
-- Ejecutar una vez en la base de datos del proyecto Lakebase.
-- La app también puede crearlo desde Configuración → "Inicializar esquema".
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- Colecciones: agrupan videos y guardan su configuración (modelos, indexación, etc.)
CREATE TABLE IF NOT EXISTS collections (
    collection_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    config        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by    TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO collections (collection_id, name, description, created_by)
VALUES ('general', 'General', 'Colección por defecto', 'system')
ON CONFLICT (collection_id) DO NOTHING;

-- Videos: un registro por video subido
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    file_name    TEXT NOT NULL,
    volume_path  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'uploaded',   -- uploaded | indexing | indexed | error
    duration_s   DOUBLE PRECISION,
    language     TEXT,
    description  TEXT,
    topics       TEXT[],
    n_segments   INT DEFAULT 0,
    n_frames     INT DEFAULT 0,
    error_msg    TEXT,
    job_run_id   BIGINT,
    uploaded_by  TEXT,
    collection_id TEXT,
    uploaded_at  TIMESTAMPTZ DEFAULT NOW(),
    indexed_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_videos_collection ON videos(collection_id);

-- Chunks vectoriales: transcripción (audio) + OCR (frames), con embedding de 1024 dims
CREATE TABLE IF NOT EXISTS video_chunks (
    id          BIGSERIAL PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,                       -- 'transcript' | 'ocr'
    seq         INT,
    start_time  DOUBLE PRECISION,
    end_time    DOUBLE PRECISION,
    text        TEXT NOT NULL,
    embedding   vector(1024),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON video_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON video_chunks(video_id);

-- Entidades extraídas por el LLM (persona/organizacion/producto/lugar/marca/…)
CREATE TABLE IF NOT EXISTS video_entities (
    id           BIGSERIAL PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    entity_type  TEXT,
    entity_value TEXT,
    mentions     INT DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_entities_video ON video_entities(video_id);

-- Capítulos / momentos clave (con timestamps)
CREATE TABLE IF NOT EXISTS video_chapters (
    id          BIGSERIAL PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    chapter_idx INT,
    start_time  DOUBLE PRECISION,
    end_time    DOUBLE PRECISION,
    title       TEXT,
    summary     TEXT
);
CREATE INDEX IF NOT EXISTS idx_chapters_video ON video_chapters(video_id);
