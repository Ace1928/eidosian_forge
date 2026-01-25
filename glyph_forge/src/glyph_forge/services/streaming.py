"""Glyph stream integration helpers."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from eidosian_core import eidosian


class GlyphStreamError(RuntimeError):
    """Raised when glyph_stream execution fails."""


def _find_glyph_stream_path() -> Path:
    """Locate the glyph_stream.py script within the repo."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "glyph_stream.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "glyph_stream.py not found. Run from the eidosian_forge repo root."
    )


def _run_subprocess(command: Sequence[str], check: bool = False) -> int:
    """Run glyph_stream command and return the exit code."""
    try:
        result = subprocess.run(command, check=check)
    except FileNotFoundError as exc:
        raise GlyphStreamError(f"Failed to run: {' '.join(command)}") from exc
    return result.returncode


def _build_common_args(
    *,
    fps: Optional[int] = None,
    scale: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    gradient_set: Optional[str] = None,
    algorithm: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    adaptive_quality: bool = True,
    border: bool = True,
    dithering: bool = False,
) -> list[str]:
    args: list[str] = []
    if fps is not None:
        args += ["--fps", str(fps)]
    if scale is not None:
        args += ["--scale", str(scale)]
    if block_width is not None:
        args += ["--block-width", str(block_width)]
    if block_height is not None:
        args += ["--block-height", str(block_height)]
    if gradient_set is not None:
        args += ["--gradient-set", gradient_set]
    if algorithm is not None:
        args += ["--algorithm", algorithm]
    if not color:
        args.append("--no-color")
    if not enhanced_edges:
        args.append("--no-enhanced-edges")
    if not adaptive_quality:
        args.append("--no-adaptive")
    if not border:
        args.append("--no-border")
    if dithering:
        args.append("--dithering")
    return args


@eidosian()
def run_glyph_stream(args: Optional[Sequence[str]] = None, check: bool = False) -> int:
    """Execute glyph_stream.py with the provided arguments."""
    path = _find_glyph_stream_path()
    command = [sys.executable, str(path)]
    if args:
        command.extend(args)
    return _run_subprocess(command, check=check)


@eidosian()
def stream_source(
    source: Union[str, Path],
    *,
    fps: Optional[int] = None,
    scale: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    gradient_set: Optional[str] = None,
    algorithm: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    adaptive_quality: bool = True,
    border: bool = True,
    dithering: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a video file or URL through glyph_stream."""
    args = [str(source)]
    args += _build_common_args(
        fps=fps,
        scale=scale,
        block_width=block_width,
        block_height=block_height,
        gradient_set=gradient_set,
        algorithm=algorithm,
        color=color,
        enhanced_edges=enhanced_edges,
        adaptive_quality=adaptive_quality,
        border=border,
        dithering=dithering,
    )
    if extra_args:
        args.extend(list(extra_args))
    return run_glyph_stream(args)


@eidosian()
def stream_webcam(
    device: int = 0,
    *,
    fps: Optional[int] = None,
    scale: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    gradient_set: Optional[str] = None,
    algorithm: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    adaptive_quality: bool = True,
    border: bool = True,
    dithering: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a webcam feed through glyph_stream."""
    args = ["--webcam", "--webcam-id", str(device)]
    args += _build_common_args(
        fps=fps,
        scale=scale,
        block_width=block_width,
        block_height=block_height,
        gradient_set=gradient_set,
        algorithm=algorithm,
        color=color,
        enhanced_edges=enhanced_edges,
        adaptive_quality=adaptive_quality,
        border=border,
        dithering=dithering,
    )
    if extra_args:
        args.extend(list(extra_args))
    return run_glyph_stream(args)


@eidosian()
def stream_youtube(
    url: str,
    *,
    fps: Optional[int] = None,
    scale: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    gradient_set: Optional[str] = None,
    algorithm: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    adaptive_quality: bool = True,
    border: bool = True,
    dithering: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a YouTube URL through glyph_stream."""
    return stream_source(
        url,
        fps=fps,
        scale=scale,
        block_width=block_width,
        block_height=block_height,
        gradient_set=gradient_set,
        algorithm=algorithm,
        color=color,
        enhanced_edges=enhanced_edges,
        adaptive_quality=adaptive_quality,
        border=border,
        dithering=dithering,
        extra_args=extra_args,
    )


@eidosian()
def stream_virtual_display(
    *,
    display_size: Optional[Union[str, tuple[int, int]]] = None,
    launch_command: Optional[Union[str, Sequence[str]]] = None,
    fps: Optional[int] = None,
    scale: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    gradient_set: Optional[str] = None,
    algorithm: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    adaptive_quality: bool = True,
    border: bool = True,
    dithering: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Stream a virtual display (optionally launching an app)."""
    args: list[str] = ["--virtual-display"]
    if display_size:
        if isinstance(display_size, tuple):
            display_size_value = f"{display_size[0]}x{display_size[1]}"
        else:
            display_size_value = str(display_size)
        args += ["--display-size", display_size_value]
    if launch_command:
        if isinstance(launch_command, (list, tuple)):
            launch_str = " ".join(shlex.quote(str(part)) for part in launch_command)
        else:
            launch_str = str(launch_command)
        args += ["--launch-app", launch_str]
    args += _build_common_args(
        fps=fps,
        scale=scale,
        block_width=block_width,
        block_height=block_height,
        gradient_set=gradient_set,
        algorithm=algorithm,
        color=color,
        enhanced_edges=enhanced_edges,
        adaptive_quality=adaptive_quality,
        border=border,
        dithering=dithering,
    )
    if extra_args:
        args.extend(list(extra_args))
    return run_glyph_stream(args)


@eidosian()
def stream_browser(
    url: str,
    *,
    browser: str = "chromium",
    kiosk: bool = True,
    display_size: Optional[Union[str, tuple[int, int]]] = (1280, 720),
    fps: Optional[int] = None,
    scale: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    gradient_set: Optional[str] = None,
    algorithm: Optional[str] = None,
    color: bool = True,
    enhanced_edges: bool = True,
    adaptive_quality: bool = True,
    border: bool = True,
    dithering: bool = False,
    browser_args: Optional[Sequence[str]] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> int:
    """Launch a browser in a virtual display and stream it as glyphs."""
    launch = [browser]
    if kiosk:
        launch += ["--kiosk", f"--app={url}"]
    else:
        launch.append(url)
    if browser_args:
        launch.extend(browser_args)
    return stream_virtual_display(
        display_size=display_size,
        launch_command=launch,
        fps=fps,
        scale=scale,
        block_width=block_width,
        block_height=block_height,
        gradient_set=gradient_set,
        algorithm=algorithm,
        color=color,
        enhanced_edges=enhanced_edges,
        adaptive_quality=adaptive_quality,
        border=border,
        dithering=dithering,
        extra_args=extra_args,
    )


__all__ = [
    "GlyphStreamError",
    "run_glyph_stream",
    "stream_source",
    "stream_webcam",
    "stream_youtube",
    "stream_virtual_display",
    "stream_browser",
]
