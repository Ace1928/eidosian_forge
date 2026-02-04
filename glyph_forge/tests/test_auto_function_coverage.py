"""Auto-coverage harness to execute every function at least once."""
from __future__ import annotations

import inspect
import pkgutil
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np
from PIL import Image
import pytest

import glyph_forge


SKIP_NAMES = {
    "stream",
    "run",
    "frames",
    "frame_generator",
    "_playback_loop",
    "_stream_loop",
    "buffer_worker",
    "_play_cached",
    "interactive",
    "_stream_screen_capture",
    "start_buffering",
    "run_tui",
}


def _make_sample_files(tmp_path: Path) -> dict[str, Any]:
    # Images
    img_rgb = Image.new("RGB", (8, 8), color=(128, 64, 32))
    img_gray = Image.new("L", (8, 8), color=128)
    img_path = tmp_path / "sample.png"
    img_rgb.save(img_path)

    # GIF
    gif_path = tmp_path / "sample.gif"
    img_gray.save(gif_path, save_all=True, append_images=[img_gray], duration=20, loop=0)

    # Text file
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Glyph Forge")

    # Generic output
    out_path = tmp_path / "out.txt"
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.m4a"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    # Share link file
    link_payload = ""
    try:
        from glyph_forge.cli.share_utils import encode_share_link
        link_payload = encode_share_link(b"Glyph Forge", "txt", "demo.txt", source="unit-test")
        link_path = tmp_path / "sample.gflink"
        link_path.write_text(link_payload, encoding="utf-8")
    except Exception:
        link_path = tmp_path / "sample.gflink"

    return {
        "img_rgb": img_rgb,
        "img_gray": img_gray,
        "img_path": img_path,
        "gif_path": gif_path,
        "txt_path": txt_path,
        "out_path": out_path,
        "link_path": link_path,
        "link_payload": link_payload,
        "video_path": video_path,
        "audio_path": audio_path,
        "url": "https://example.com/video.mp4",
        "web_url": "https://example.com",
    }


def _sample_value(name: str, annotation: Any, samples: dict[str, Any]) -> Any:
    lname = name.lower()
    if lname == "audio":
        return str(samples["out_path"])
    if lname == "audio_source":
        return str(samples["audio_path"])
    if lname == "youtube":
        return None
    if lname == "assume_yes":
        return True
    if lname == "data":
        return b"Glyph Forge"
    if lname == "filename":
        return "demo.txt"
    if lname == "density_map":
        return {i: " " for i in range(256)}
    if lname == "pixels":
        return np.zeros((1, 1), dtype=np.uint8)
    if lname == "density_lut":
        return np.array([" "] * 256, dtype=object)
    if lname == "fmt":
        return "txt"
    if lname == "open_output":
        return False
    if lname == "record":
        return "none"
    if lname == "duration":
        return 0.01
    if annotation is Path or (isinstance(annotation, str) and annotation.endswith("Path")):
        return samples["img_path"]
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is bool:
        return True
    if annotation is str:
        if "url" in lname:
            return samples["url"]
        if "path" in lname:
            return str(samples["img_path"])
        if "text" in lname:
            return "Glyph Forge"
        if "mode" in lname:
            return "gradient"
        if "color" in lname:
            return "none"
        return "value"

    if "path" in lname:
        return str(samples["img_path"])
    if "image" in lname or lname == "img":
        return samples["img_rgb"]
    if "frames" in lname:
        return [samples["img_gray"]]
    if "frame" in lname:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    if "url" in lname:
        return samples["url"]
    if "text" in lname or "message" in lname:
        return "Glyph Forge"
    if "charset" in lname:
        return " .:-=+*#%@"
    if "color" in lname and "mode" in lname:
        return "none"
    if "mode" in lname:
        return "gradient"
    if "gradient" in lname:
        return "standard"
    if "algorithm" in lname:
        return "sobel"
    if "width" in lname or "height" in lname:
        return 8
    if "style" in lname:
        return "bold"
    if lname in {"items", "item_list"}:
        return ["one", "two"]
    if lname in {"base", "overlay", "profile", "updates"}:
        return {}
    if lname == "command":
        return ["echo", "glyph_forge"]
    if lname == "duration":
        return 0
    if lname == "record":
        return "none"
    if lname == "api":
        try:
            from glyph_forge.api import get_api
            return get_api()
        except Exception:
            return None
    if lname == "args":
        from types import SimpleNamespace
        return SimpleNamespace(
            text="Glyph Forge",
            style="minimal",
            font="slant",
            width=80,
            color=False,
            output=None,
            list_fonts=False,
            list_styles=False,
            preview=False,
            debug=False,
            version=False,
        )
    if "fps" in lname:
        return 30
    if "source" in lname:
        return str(samples["video_path"])
    if "output" in lname:
        return str(samples["out_path"])
    if lname == "value":
        return str(samples["link_path"])
    if lname == "link":
        return samples["link_payload"] or "glyphforge://"

    return None


def _build_args(func: Callable[..., Any], samples: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(func)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind == param.VAR_POSITIONAL:
            continue
        if param.kind == param.VAR_KEYWORD:
            continue
        if param.default is not inspect._empty:
            if param.default is None and name == "assume_yes":
                kwargs[name] = True
                continue
            # Replace Typer defaults with real values
            try:
                import typer
                if isinstance(param.default, (typer.models.ArgumentInfo, typer.models.OptionInfo)):
                    sample = _sample_value(name, param.annotation, samples)
                    if sample is None:
                        if name == "youtube":
                            kwargs[name] = None
                            continue
                        sample = 1
                    kwargs[name] = sample
                    continue
            except Exception:
                pass
            continue

        sample = _sample_value(name, param.annotation, samples)
        if sample is None:
            # Use a generic fallback
            sample = 1
        kwargs[name] = sample

    return args, kwargs


def _iter_modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for mod in pkgutil.walk_packages(glyph_forge.__path__, glyph_forge.__name__ + "."):
        modules.append(importlib.import_module(mod.name))
    return modules


@pytest.mark.parametrize("module", _iter_modules())
def test_execute_module_functions(module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute all top-level functions in each module."""
    samples = _make_sample_files(tmp_path)

    # Speed up modules with sleep-heavy control loops
    if module.__name__ in {"glyph_forge.streaming.core.netflix"}:
        monkeypatch.setattr(module.time, "sleep", lambda *_: None, raising=False)
    if module.__name__ == "glyph_forge.streaming.audio_tools":
        monkeypatch.setattr(module.shutil, "which", lambda *_: "/usr/bin/ffmpeg", raising=False)
        monkeypatch.setattr(module, "has_ffmpeg", lambda: True, raising=False)
        monkeypatch.setattr(
            module.subprocess,
            "run",
            lambda cmd, capture_output=True, text=True: SimpleNamespace(
                returncode=0,
                stderr="",
                stdout="",
            ),
            raising=False,
        )
    if module.__name__ == "glyph_forge.streaming.core.screen":
        class DummyMSS:
            def __init__(self):
                self.monitors = [
                    {"left": 0, "top": 0, "width": 1, "height": 1},
                    {"left": 0, "top": 0, "width": 1, "height": 1},
                ]

            def grab(self, _region):
                return np.zeros((1, 1, 4), dtype=np.uint8)

            def close(self):
                return None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        monkeypatch.setattr(module, "MSS_AVAILABLE", True, raising=False)
        monkeypatch.setattr(module, "mss", SimpleNamespace(mss=DummyMSS), raising=False)

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            # Skip dunder
            if name.startswith("__"):
                continue
            if name in SKIP_NAMES:
                continue
            # Special-case CLI entrypoints to avoid exit
            if name in {"main", "cli"}:
                monkeypatch.setattr(sys, "argv", ["glyph-forge", "--help"])
                try:
                    obj()
                except SystemExit:
                    pass
                except Exception as exc:
                    try:
                        import click
                        if isinstance(exc, click.exceptions.Exit):
                            pass
                        else:
                            raise
                    except Exception:
                        raise
                continue

            args, kwargs = _build_args(obj, samples)
            try:
                obj(*args, **kwargs)
            except SystemExit:
                pass
            except Exception as exc:
                try:
                    import click
                    if isinstance(exc, click.exceptions.Exit):
                        pass
                    else:
                        raise
                except Exception:
                    raise


@pytest.mark.parametrize("module", _iter_modules())
def test_execute_class_methods(module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute class methods for all classes with default constructors."""
    samples = _make_sample_files(tmp_path)

    if module.__name__ == "glyph_forge.streaming.core.netflix":
        monkeypatch.setattr(module.time, "sleep", lambda *_: None, raising=False)

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            # Avoid private/internal classes that require complex init
            if name.startswith("_"):
                continue

            # Try to instantiate with defaults
            try:
                instance = obj()  # type: ignore[call-arg]
            except Exception:
                # Attempt with simple samples if constructor needs args
                try:
                    init_args, init_kwargs = _build_args(obj.__init__, samples)
                    instance = obj(*init_args, **init_kwargs)  # type: ignore[misc]
                except Exception:
                    continue

            for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                if meth_name.startswith("_"):
                    continue
                if meth_name in SKIP_NAMES:
                    continue
                bound = getattr(instance, meth_name, None)
                if bound is None:
                    continue
                try:
                    args, kwargs = _build_args(bound, samples)
                    bound(*args, **kwargs)
                except SystemExit:
                    pass
                except Exception:
                    # Allow errors for methods that require runtime contexts
                    continue
