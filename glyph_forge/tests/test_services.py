"""Tests for high-level service functions."""

from pathlib import Path
from PIL import Image

from glyph_forge.services import (
    text_to_banner,
    text_to_glyph,
    video_to_glyph_frames,
    stream_source,
    stream_webcam,
    stream_browser,
)
from unittest import mock


def test_text_to_banner_basic() -> None:
    """Ensure ``text_to_banner`` returns non-empty glyph art."""
    result = text_to_banner("Forge")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_text_to_glyph() -> None:
    """Ensure ``text_to_glyph`` mirrors banner service output."""
    result = text_to_glyph("Forge")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_video_to_glyph_frames(tmp_path: Path) -> None:
    """Verify ``video_to_glyph_frames`` processes GIF frames."""
    # Create simple 2-frame GIF
    frame1 = Image.new("L", (8, 8), color=0)
    frame2 = Image.new("L", (8, 8), color=255)
    gif_path = tmp_path / "two.gif"
    frame1.save(gif_path, save_all=True, append_images=[frame2], duration=20, loop=0)

    frames = video_to_glyph_frames(str(gif_path), width=8)
    assert len(frames) == 2
    assert all(isinstance(f, str) for f in frames)


def test_stream_source_builds_args() -> None:
    """Verify streaming helper builds glyph_stream arguments."""
    with mock.patch("glyph_forge.services.streaming.run_glyph_stream") as run_stream:
        stream_source(
            "video.mp4",
            fps=12,
            scale=2,
            block_width=6,
            block_height=8,
            gradient_set="ascii",
            algorithm="sobel",
            color=False,
            enhanced_edges=False,
            adaptive_quality=False,
            border=False,
            dithering=True,
            extra_args=["--debug"],
        )

        run_stream.assert_called_once()
        args = run_stream.call_args.args[0]
        assert args[0] == "video.mp4"
        assert "--fps" in args and "12" in args
        assert "--scale" in args and "2" in args
        assert "--block-width" in args and "6" in args
        assert "--block-height" in args and "8" in args
        assert "--gradient-set" in args and "ascii" in args
        assert "--algorithm" in args and "sobel" in args
        assert "--no-color" in args
        assert "--no-enhanced-edges" in args
        assert "--no-adaptive" in args
        assert "--no-border" in args
        assert "--dithering" in args
        assert "--debug" in args


def test_stream_webcam_builds_args() -> None:
    """Verify webcam streaming arguments."""
    with mock.patch("glyph_forge.services.streaming.run_glyph_stream") as run_stream:
        stream_webcam(device=2, fps=10, scale=1)

        run_stream.assert_called_once()
        args = run_stream.call_args.args[0]
        assert args[:3] == ["--webcam", "--webcam-id", "2"]
        assert "--fps" in args and "10" in args
        assert "--scale" in args and "1" in args


def test_stream_browser_builds_args() -> None:
    """Verify browser streaming builds launch arguments."""
    with mock.patch("glyph_forge.services.streaming.run_glyph_stream") as run_stream:
        stream_browser("https://example.com", fps=15, scale=2, kiosk=True)

        run_stream.assert_called_once()
        args = run_stream.call_args.args[0]
        assert "--virtual-display" in args
        assert "--launch-app" in args
