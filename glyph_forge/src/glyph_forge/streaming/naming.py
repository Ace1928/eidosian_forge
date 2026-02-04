"""Naming and metadata helpers for streaming output."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json
import shutil
import subprocess
import re
from urllib.parse import urlparse, parse_qs


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\\.be/)([a-zA-Z0-9_-]{11})")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif", ".apng"}


def slugify(text: str, fallback: str = "glyph_forge") -> str:
    if not text:
        return fallback
    text = text.strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or fallback


def _youtube_id_from_url(source: str) -> Optional[str]:
    match = YOUTUBE_ID_RE.search(source)
    if match:
        return match.group(1)
    try:
        parsed = urlparse(source)
        if parsed.netloc.endswith("youtube.com"):
            qs = parse_qs(parsed.query)
            vid = qs.get("v", [None])[0]
            if vid:
                return vid
    except Exception:
        return None
    return None


def derive_title(source: str, title: Optional[str] = None) -> str:
    if title:
        return title
    yt_id = _youtube_id_from_url(source)
    if yt_id:
        return f"youtube_{yt_id}"
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        tail = Path(parsed.path).stem or parsed.netloc
        return tail or "glyph_forge"
    return Path(source).stem or "glyph_forge"


def build_output_path(
    source: str,
    title: Optional[str] = None,
    output_dir: Optional[Path] = None,
    ext: str = "mp4",
    output: Optional[str | Path] = None,
    overwrite: bool = False,
) -> Path:
    if output:
        return Path(output)
    base = slugify(derive_title(source, title))
    out_dir = output_dir or (Path.cwd() / "glyph_forge_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{base}.{ext}"
    if overwrite or not candidate.exists():
        return candidate
    idx = 1
    while True:
        path = out_dir / f"{base}_{idx:03d}.{ext}"
        if not path.exists():
            return path
        idx += 1


def build_metadata(
    source: str,
    output_path: Path,
    title: Optional[str] = None,
    info: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "source": source,
        "title": title or derive_title(source, title),
        "output_path": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if info is not None:
        for key in ("video_url", "audio_url", "duration", "fps", "width", "height", "format", "is_live"):
            value = getattr(info, key, None)
            if value is not None:
                meta[key] = value
    media = _probe_video_metadata(output_path)
    if media:
        meta["output_media"] = media
    if extra:
        meta.update(extra)
    return meta


def _probe_video_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.suffix.lower() not in VIDEO_EXTS:
        return {}
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,bit_rate",
        "-show_entries", "format=duration,size",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}
    if not streams:
        return {}
    stream = streams[0]
    fps = None
    rate = stream.get("avg_frame_rate")
    if rate and isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        try:
            if float(den) != 0:
                fps = float(num) / float(den)
        except ValueError:
            fps = None
    return {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "bit_rate": stream.get("bit_rate"),
        "duration": float(fmt.get("duration")) if fmt.get("duration") else None,
        "size": int(fmt.get("size")) if fmt.get("size") else None,
    }


def write_metadata(metadata: Dict[str, Any], output_path: Path) -> Path:
    meta_path = output_path.with_suffix(".metadata.json")
    meta_path.write_text(
        __import__("json").dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return meta_path
