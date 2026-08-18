"""Manejo de archivos de video en UC Volumes + disparo del Job de indexación."""
import io
import time
from typing import Optional

import config

_JOB_STATE_CACHE: dict = {}


def upload_video(video_id: str, file_name: str, data: bytes,
                 collection_id: str = "general") -> str:
    """Sube el video al Volume, en la subcarpeta de la colección. Devuelve la ruta."""
    w = config.get_workspace_client()
    safe = file_name.replace("/", "_").replace(" ", "_")
    dest = f"{config.cfg('uploads_volume').rstrip('/')}/{collection_id}/{video_id}__{safe}"
    w.files.upload(dest, io.BytesIO(data), overwrite=True)
    return dest


def test_volume(volume_path: str) -> dict:
    """Verifica acceso de escritura al Volume: escribe y borra un archivo temporal."""
    w = config.get_workspace_client()
    probe = f"{volume_path.rstrip('/')}/.vi_write_test"
    w.files.upload(probe, io.BytesIO(b"ok"), overwrite=True)
    w.files.delete(probe)
    return {"ok": True, "path": volume_path}


def download_video(volume_path: str) -> bytes:
    """Descarga los bytes del video desde el Volume (para reproducción)."""
    w = config.get_workspace_client()
    resp = w.files.download(volume_path)
    return resp.contents.read()


def delete_video_file(volume_path: str):
    w = config.get_workspace_client()
    try:
        w.files.delete(volume_path)
    except Exception:
        pass


def _set_job_email(w, job_id: int, email: str):
    """Configura (o limpia) las notificaciones por correo del Job antes de dispararlo.

    email_notifications se define a nivel Job (no por-run), así que lo actualizamos
    dinámicamente con el correo de la colección justo antes de run_now. Best-effort:
    si falla (p.ej. permisos), la indexación continúa igual.
    """
    from databricks.sdk.service import jobs as j
    recipients = [email] if email else []
    notif = j.JobEmailNotifications(on_success=recipients, on_failure=recipients,
                                    no_alert_for_skipped_runs=False)
    w.jobs.update(job_id=job_id, new_settings=j.JobSettings(email_notifications=notif))


def trigger_indexing(video_id: str, volume_path: str,
                     whisper_model: str = "small", frame_interval: int = 5,
                     language: str = "auto") -> int:
    """Lanza el Job de indexación con los modelos/Lakebase configurados. Devuelve run_id."""
    w = config.get_workspace_client()
    job_id = config.indexer_job_id()
    # notificación por correo de la colección activa (best-effort)
    try:
        _set_job_email(w, job_id, str(config.cfg("notify_email") or "").strip())
    except Exception:
        pass
    run = w.jobs.run_now(
        job_id=job_id,
        notebook_params={
            "video_id": video_id,
            "volume_path": volume_path,
            "whisper_model": whisper_model,
            "frame_interval": str(frame_interval),
            "language": language,
            "frames_volume": config.cfg("frames_volume"),
            # modelos y coordenadas Lakebase efectivos (configurables en la app)
            "embedding_endpoint": config.cfg("embedding_endpoint"),
            "llm_endpoint": config.cfg("llm_endpoint"),
            "lb_project": config.cfg("lb_project"),
            "lb_branch": config.cfg("lb_branch"),
            "lb_endpoint": config.cfg("lb_endpoint"),
            "lb_host": config.resolve_pg_host(),
            "lb_db": config.cfg("lb_database"),
        },
    )
    return run.run_id


def run_status(run_id: int) -> dict:
    """Estado del run del Job (life_cycle + result state + url)."""
    w = config.get_workspace_client()
    r = w.jobs.get_run(run_id)
    life = r.state.life_cycle_state.value if r.state and r.state.life_cycle_state else None
    result = r.state.result_state.value if r.state and r.state.result_state else None
    return {"life_cycle": life, "result": result, "run_page_url": r.run_page_url}
