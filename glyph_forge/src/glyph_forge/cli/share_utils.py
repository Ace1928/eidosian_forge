"""Sharing utilities for Glyph Forge CLI."""
from __future__ import annotations

import base64
import gzip
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from ..renderers import HTMLRenderer, SVGRenderer
from ..streaming.naming import derive_title, slugify

LINK_PREFIX = "glyphforge://"


def build_share_path(
    source: str | None,
    fmt: str,
    output: str | None = None,
    title: str | None = None,
) -> Path:
    """Build a deterministic share output path."""
    if output:
        return Path(output)
    stem = slugify(derive_title(source or "glyph_forge", title))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / "glyph_forge_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{timestamp}.{fmt}"


def render_share(content: str, fmt: str) -> str:
    """Render string content into a shareable text format."""
    if not isinstance(fmt, str):
        return str(content)
    if fmt == "txt":
        return str(content)
    if fmt == "html":
        if "<pre" in content or "<html" in content:
            return content
        matrix = _content_to_matrix(content)
        return HTMLRenderer().render(matrix)
    if fmt == "svg":
        matrix = _content_to_matrix(content)
        return SVGRenderer().render(matrix)
    raise ValueError(f"Unsupported share format: {fmt}")


def write_share(content: str, fmt: str, path: Path) -> None:
    """Write share output to file (supports binary formats)."""
    path = Path(path)
    if not isinstance(fmt, str):
        fmt = "txt"
    fmt = fmt.lower()
    if fmt in {"txt", "html", "svg"}:
        path.write_text(render_share(content, fmt), encoding="utf-8")
        return
    if fmt in {"png", "gif"}:
        if fmt == "gif" and _is_frame_sequence(content):
            _render_sequence_gif(content, path)
            return
        if _contains_ansi(content):
            _render_ansi_image(content, path, fmt)
        else:
            _render_text_image(content, path, fmt)
        return
    # Fallback to text
    path.write_text(render_share(content, "txt"), encoding="utf-8")


def export_video_share(video_path: Path, fmt: str, output_path: Path) -> bool:
    """Export glyph video to shareable formats."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        return False
    if fmt == "mp4":
        output_path.write_bytes(video_path.read_bytes())
        return True
    if fmt == "png":
        return _ffmpeg_extract_frame(video_path, output_path)
    if fmt == "gif":
        return _ffmpeg_extract_gif(video_path, output_path)
    if fmt == "apng":
        return _ffmpeg_extract_apng(video_path, output_path)
    if fmt == "webm":
        return _ffmpeg_extract_webm(video_path, output_path)
    return False


def try_open_path(path: Path) -> bool:
    """Best-effort open a file with system default application."""
    opener = None
    if shutil.which("xdg-open"):
        opener = ["xdg-open", str(path)]
    elif shutil.which("open"):
        opener = ["open", str(path)]
    if opener:
        subprocess.run(opener, check=False)
        return True
    return False


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using available mechanisms."""
    try:
        import pyperclip  # type: ignore
        pyperclip.copy(text)
        return True
    except Exception:
        pass

    candidates: list[list[str]] = []
    if shutil.which("pbcopy"):
        candidates.append(["pbcopy"])
    if shutil.which("wl-copy"):
        candidates.append(["wl-copy"])
    if shutil.which("xclip"):
        candidates.append(["xclip", "-selection", "clipboard"])
    for cmd in candidates:
        try:
            subprocess.run(cmd, input=text.encode(), check=False)
            return True
        except Exception:
            continue
    return False


def encode_share_link(
    data: bytes,
    fmt: str,
    filename: str,
    source: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Encode data into a portable share link."""
    compression = None
    compressed = data
    if len(data) > 2048:
        trial = gzip.compress(data)
        if len(trial) < len(data):
            compressed = trial
            compression = "gzip"

    payload = {
        "version": 1,
        "format": fmt,
        "filename": filename,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compression": compression,
        "data_b64": base64.urlsafe_b64encode(compressed).decode("ascii"),
    }
    if metadata:
        payload["metadata"] = metadata
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return LINK_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii")


def decode_share_link(link: str) -> Tuple[dict, bytes]:
    """Decode a share link back into payload and raw bytes."""
    raw = link.strip()
    if raw.startswith(LINK_PREFIX):
        raw = raw[len(LINK_PREFIX):]
    blob = base64.urlsafe_b64decode(raw.encode("ascii"))
    payload = json.loads(blob.decode("utf-8"))
    data = base64.urlsafe_b64decode(payload["data_b64"].encode("ascii"))
    if payload.get("compression") == "gzip":
        data = gzip.decompress(data)
    return payload, data


def load_share_link(value: str) -> Tuple[dict, bytes]:
    """Load a share link from a raw string or a file path."""
    path = Path(value)
    if path.exists():
        return decode_share_link(path.read_text(encoding="utf-8").strip())
    return decode_share_link(value)


def _content_to_matrix(content: str) -> list[list[str]]:
    if not isinstance(content, str):
        content = str(content)
    lines = content.splitlines() or [content]
    return [list(line) for line in lines]


def _render_text_image(content: str, path: Path, fmt: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not isinstance(content, str):
        content = str(content)
    lines = content.splitlines() or [content]
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 14)
            break
    if font is None:
        font = ImageFont.load_default()

    max_width = max(len(line) for line in lines)
    char_w, char_h = font.getbbox("M")[2:4]
    img_w = max(1, max_width * char_w)
    img_h = max(1, len(lines) * (char_h + 2))

    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(255, 255, 255))
        y += char_h + 2

    if fmt == "png":
        img.save(path, format="PNG")
    elif fmt == "gif":
        img.save(path, format="GIF", save_all=True, duration=100, loop=0)


def _render_ansi_image(content: str, path: Path, fmt: str) -> None:
    from PIL import Image
    from glyph_forge.streaming.core.recorder import GlyphRecorder, RecorderConfig

    import tempfile
    import uuid
    temp_path = Path(tempfile.gettempdir()) / f"glyph_render_{uuid.uuid4().hex}.mp4"
    try:
        recorder = GlyphRecorder(RecorderConfig(output_path=temp_path))
        frame = recorder.render_to_image(content)
        img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    except Exception:
        # Keep share export functional when optional video backends are unavailable.
        _render_text_image(content, path, fmt)
        return
    if fmt == "png":
        img.save(path, format="PNG")
    elif fmt == "gif":
        img.save(path, format="GIF", save_all=True, duration=100, loop=0)


def _render_sequence_gif(frames: Sequence[str], path: Path, duration_ms: int = 80) -> None:
    from PIL import Image

    frames_list = list(frames)
    if not frames_list:
        return
    if any(_contains_ansi(frame) for frame in frames_list):
        imgs = [_ansi_frame_to_image(frame) for frame in frames_list]
    else:
        imgs = [_text_frame_to_image(frame) for frame in frames_list]
    imgs = [img.convert("P", palette=Image.ADAPTIVE) for img in imgs]
    imgs[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _contains_ansi(content: str) -> bool:
    if not isinstance(content, str):
        return False
    return "\033[" in content


def _is_frame_sequence(content: object) -> bool:
    if isinstance(content, str):
        return False
    return isinstance(content, Sequence)


def _text_frame_to_image(content: str):
    from PIL import Image, ImageDraw, ImageFont

    if not isinstance(content, str):
        content = str(content)
    lines = content.splitlines() or [content]
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 14)
            break
    if font is None:
        font = ImageFont.load_default()

    max_width = max(len(line) for line in lines)
    char_w, char_h = font.getbbox("M")[2:4]
    img_w = max(1, max_width * char_w)
    img_h = max(1, len(lines) * (char_h + 2))

    img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(255, 255, 255))
        y += char_h + 2
    return img


def _ansi_frame_to_image(content: str):
    from PIL import Image
    from glyph_forge.streaming.core.recorder import GlyphRecorder, RecorderConfig

    import tempfile
    import uuid
    temp_path = Path(tempfile.gettempdir()) / f"glyph_render_{uuid.uuid4().hex}.mp4"
    try:
        recorder = GlyphRecorder(RecorderConfig(output_path=temp_path))
        frame = recorder.render_to_image(content)
        return Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    except Exception:
        return _text_frame_to_image(content)


def _ffmpeg_extract_frame(video_path: Path, output_path: Path) -> bool:
    import subprocess
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def _ffmpeg_extract_gif(video_path: Path, output_path: Path) -> bool:
    import subprocess
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "fps=12,scale=640:-1:flags=lanczos",
        "-loop", "0",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def _ffmpeg_extract_apng(video_path: Path, output_path: Path) -> bool:
    import subprocess
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "fps=12,scale=640:-1:flags=lanczos",
        "-plays", "0",
        "-f", "apng",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def _ffmpeg_extract_webm(video_path: Path, output_path: Path) -> bool:
    import subprocess
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-c:v", "libvpx-vp9",
        "-b:v", "2M",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()
