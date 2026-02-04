"""Benchmark coverage harness for glyph_forge functions."""
from __future__ import annotations

import inspect
import os
import pkgutil
import importlib
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import numpy as np
from PIL import Image
import pytest

import glyph_forge

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

SKIP_BENCHMARK_NAMES = {
    "run_tui",
    "interactive",
}
SKIP_BENCHMARK_METHODS = {
    "start_buffering",
}
BENCH_SCOPE = os.environ.get("GLYPH_FORGE_BENCHMARK_SCOPE", "core").lower()
BENCH_LIMIT = int(os.environ.get("GLYPH_FORGE_BENCHMARK_LIMIT", "200"))
BENCH_ROUNDS = int(os.environ.get("GLYPH_FORGE_BENCHMARK_ROUNDS", "1"))
BENCH_ITERS = int(os.environ.get("GLYPH_FORGE_BENCHMARK_ITERS", "1"))


def _include_module(mod_name: str) -> bool:
    if BENCH_SCOPE == "all":
        return True
    return mod_name.startswith(
        (
            "glyph_forge.core",
            "glyph_forge.renderers",
            "glyph_forge.services",
            "glyph_forge.streaming",
            "glyph_forge.transformers",
        )
    )


def _make_samples(tmp_path: Path) -> dict[str, Any]:
    img_rgb = Image.new("RGB", (8, 8), color=(128, 64, 32))
    img_path = tmp_path / "sample.png"
    img_rgb.save(img_path)
    gif_path = tmp_path / "sample.gif"
    img_rgb.convert("L").save(gif_path, save_all=True, append_images=[img_rgb.convert("L")])
    return {
        "img": img_rgb,
        "img_path": img_path,
        "gif_path": gif_path,
        "frame": np.zeros((8, 8, 3), dtype=np.uint8),
        "url": "https://example.com/video.mp4",
        "text": "Glyph Forge",
    }


def _sample_value(name: str, annotation: Any, samples: dict[str, Any]) -> Any:
    lname = name.lower()
    if lname == "record":
        return "none"
    if lname == "duration":
        return 0.01
    if lname == "timeout":
        return 0.001
    if annotation is Path:
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
            return samples["text"]
        if "mode" in lname:
            return "gradient"
        if "color" in lname:
            return "none"
        return "value"

    if "path" in lname:
        return str(samples["img_path"])
    if "image" in lname or lname == "img":
        return samples["img"]
    if "url" in lname:
        return samples["url"]
    if "text" in lname:
        return samples["text"]
    if "width" in lname or "height" in lname:
        return 8
    if "gradient" in lname:
        return "standard"
    if "algorithm" in lname:
        return "sobel"
    if "fps" in lname:
        return 30
    if lname in {"items", "item_list"}:
        return ["one", "two"]
    if lname in {"base", "overlay", "profile", "updates"}:
        return {}
    if lname == "command":
        return ["echo", "glyph_forge"]
    if "frames" in lname:
        return [samples["img"]]
    if "frame" in lname:
        return samples["frame"]
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
    return None


def _build_args(func: Callable[..., Any], samples: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            try:
                import typer
                if isinstance(param.default, (typer.models.ArgumentInfo, typer.models.OptionInfo)):
                    sample = _sample_value(name, param.annotation, samples)
                    if sample is None:
                        sample = 1
                    kwargs[name] = sample
                    continue
            except Exception:
                pass
            continue
        sample = _sample_value(name, param.annotation, samples)
        if sample is None:
            sample = 1
        kwargs[name] = sample
    return [], kwargs


def _iter_modules() -> list[str]:
    modules: list[str] = []
    for mod in pkgutil.walk_packages(glyph_forge.__path__, glyph_forge.__name__ + "."):
        if _include_module(mod.name):
            modules.append(mod.name)
    return modules


def _iter_function_targets() -> list[Tuple[str, str]]:
    targets: list[Tuple[str, str]] = []
    for mod_name in _iter_modules():
        module = importlib.import_module(mod_name)
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                if name.startswith("__"):
                    continue
                if name in SKIP_BENCHMARK_NAMES:
                    continue
                targets.append((mod_name, name))
    if BENCH_LIMIT > 0:
        return targets[:BENCH_LIMIT]
    return targets


def _iter_method_targets() -> list[Tuple[str, str, str]]:
    targets: list[Tuple[str, str, str]] = []
    for mod_name in _iter_modules():
        module = importlib.import_module(mod_name)
        for cls_name, cls in inspect.getmembers(module):
            if inspect.isclass(cls) and cls.__module__ == module.__name__:
                if cls_name.startswith("_"):
                    continue
                for meth_name, meth in inspect.getmembers(cls, predicate=inspect.isfunction):
                    if meth_name.startswith("_"):
                        continue
                    if meth_name in SKIP_BENCHMARK_METHODS:
                        continue
                    targets.append((mod_name, cls_name, meth_name))
    if BENCH_LIMIT > 0:
        return targets[:BENCH_LIMIT]
    return targets


@pytest.mark.parametrize("module_name, func_name", _iter_function_targets())
def test_benchmark_function(module_name: str, func_name: str, benchmark, tmp_path: Path) -> None:
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    samples = _make_samples(tmp_path)
    args, kwargs = _build_args(func, samples)
    restore = []
    if module_name.endswith("streaming.core.netflix"):
        if hasattr(module, "time"):
            restore.append(("time.sleep", module.time.sleep))
            module.time.sleep = lambda *_args, **_kwargs: None  # type: ignore[assignment]
        if hasattr(module, "NetflixCapture"):
            restore.append(("NetflixCapture.start", module.NetflixCapture.start))
            restore.append(("NetflixCapture.stop", module.NetflixCapture.stop))
            module.NetflixCapture.start = lambda *_args, **_kwargs: True  # type: ignore[assignment]
            module.NetflixCapture.stop = lambda *_args, **_kwargs: Path("netflix.mp4")  # type: ignore[assignment]

    def _safe_call():
        try:
            return func(*args, **kwargs)
        except SystemExit:
            return None
        except Exception:
            return None

    try:
        if hasattr(benchmark, "pedantic"):
            benchmark.pedantic(_safe_call, iterations=BENCH_ITERS, rounds=BENCH_ROUNDS, warmup_rounds=0)
        else:
            benchmark(_safe_call)
    finally:
        for name, original in reversed(restore):
            if name == "time.sleep":
                module.time.sleep = original  # type: ignore[assignment]
            elif name == "NetflixCapture.start":
                module.NetflixCapture.start = original  # type: ignore[assignment]
            elif name == "NetflixCapture.stop":
                module.NetflixCapture.stop = original  # type: ignore[assignment]


@pytest.mark.parametrize("module_name, cls_name, meth_name", _iter_method_targets())
def test_benchmark_method(module_name: str, cls_name: str, meth_name: str, benchmark, tmp_path: Path) -> None:
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    samples = _make_samples(tmp_path)
    restore = []

    try:
        instance = cls()  # type: ignore[call-arg]
    except Exception:
        try:
            init_args, init_kwargs = _build_args(cls.__init__, samples)
            instance = cls(*init_args, **init_kwargs)  # type: ignore[misc]
        except Exception:
            pytest.skip("Unable to instantiate class for benchmark")

    if module_name.endswith("streaming.core.netflix"):
        if hasattr(module, "time"):
            restore.append(("time.sleep", module.time.sleep))
            module.time.sleep = lambda *_args, **_kwargs: None  # type: ignore[assignment]
        if hasattr(module, "NetflixCapture"):
            restore.append(("NetflixCapture.start", module.NetflixCapture.start))
            restore.append(("NetflixCapture.stop", module.NetflixCapture.stop))
            module.NetflixCapture.start = lambda *_args, **_kwargs: True  # type: ignore[assignment]
            module.NetflixCapture.stop = lambda *_args, **_kwargs: Path("netflix.mp4")  # type: ignore[assignment]

    if module_name.endswith("streaming.core.buffer") and cls_name == "AdaptiveBuffer":
        if hasattr(instance, "_frames"):
            instance._frames.append("frame")  # type: ignore[attr-defined]
        if hasattr(instance, "_buffering_complete"):
            instance._buffering_complete = True  # type: ignore[attr-defined]

    meth = getattr(instance, meth_name)
    args, kwargs = _build_args(meth, samples)

    def _safe_call():
        try:
            return meth(*args, **kwargs)
        except SystemExit:
            return None
        except Exception:
            return None

    try:
        if hasattr(benchmark, "pedantic"):
            benchmark.pedantic(_safe_call, iterations=BENCH_ITERS, rounds=BENCH_ROUNDS, warmup_rounds=0)
        else:
            benchmark(_safe_call)
    finally:
        for name, original in reversed(restore):
            if name == "time.sleep":
                module.time.sleep = original  # type: ignore[assignment]
            elif name == "NetflixCapture.start":
                module.NetflixCapture.start = original  # type: ignore[assignment]
            elif name == "NetflixCapture.stop":
                module.NetflixCapture.stop = original  # type: ignore[assignment]
