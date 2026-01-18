import os
import sys
from types import SimpleNamespace
from pathlib import Path
import importlib.machinery
import importlib.util
import builtins

CTX_PATH = Path(__file__).resolve().parents[1] / "glyph_stream_context.py"


def load_module(name="glyph_stream_context", fake_numpy=True):
    if fake_numpy:
        sys.modules["numpy"] = SimpleNamespace(
            __config__=SimpleNamespace(show=lambda: "mkl")
        )
    loader = importlib.machinery.SourceFileLoader(name, str(CTX_PATH))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_system_context_basic():
    ctx_mod = load_module("glyph_stream_context_basic")
    ctx = ctx_mod.SystemContext(check_network=False)
    assert "platform" in ctx.attributes
    assert "can_display_color" in ctx.capabilities


def test_color_support_env(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_color")
    ctx = ctx_mod.SystemContext(check_network=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert ctx._detect_color_support({"platform": "Linux"}) is False
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert ctx._detect_color_support({"platform": "Linux"}) is True
    monkeypatch.delenv("COLORTERM", raising=False)


def test_color_support_windows(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_win")
    ctx = ctx_mod.SystemContext(check_network=False)

    class DummyWin:
        build = 19041

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("ANSICON", raising=False)
    monkeypatch.setattr(
        ctx_mod.sys, "getwindowsversion", lambda: DummyWin(), raising=False
    )
    assert ctx._detect_color_support({"platform": "Windows"}) is True

    class DummyWinOld:
        build = 10000

    monkeypatch.setattr(
        ctx_mod.sys, "getwindowsversion", lambda: DummyWinOld(), raising=False
    )
    monkeypatch.setenv("ANSICON", "1")
    assert ctx._detect_color_support({"platform": "Windows"}) is True
    monkeypatch.delenv("ANSICON", raising=False)


def test_network_check(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_net")
    monkeypatch.setattr(
        ctx_mod.SystemContext, "_check_network_connection", lambda self: True
    )
    ctx = ctx_mod.SystemContext(check_network=True)
    assert ctx.attributes.get("network_connected") is True


def test_hardware_accel_torch(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("EIDOSIAN_ENABLE_TORCH", "1")

    ctx_mod = load_module("glyph_stream_context_torch")
    ctx = ctx_mod.SystemContext(check_network=False)
    assert ctx.attributes.get("has_cuda") is True


def test_hardware_accel_darwin():
    ctx_mod = load_module("glyph_stream_context_darwin")
    ctx = ctx_mod.SystemContext(check_network=False)
    assert (
        ctx._detect_hardware_acceleration({"platform": "Darwin"})["has_metal"] is True
    )


def test_import_errors(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_import", fake_numpy=False)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("psutil", "numpy", "torch"):
            raise ImportError("nope")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ctx = ctx_mod.SystemContext(check_network=False)
    assert ctx.attributes.get("memory_detection") == "unavailable"


def test_constraints_and_paths():
    ctx_mod = load_module("glyph_stream_context_constraints")
    ctx = ctx_mod.SystemContext(check_network=False)
    ctx.attributes["terminal_width"] = 50
    ctx.attributes["terminal_height"] = 10
    ctx.attributes["cpu_count"] = 1
    ctx.attributes["memory_available"] = 1 * 1024 * 1024 * 1024
    ctx.attributes["has_cuda"] = False
    ctx.attributes["has_metal"] = False
    ctx.capabilities["can_display_color"] = False
    ctx.capabilities["can_display_unicode"] = False
    ctx.constraints = ctx._detect_constraints()
    ctx.optimization_paths = ctx._generate_optimization_paths()
    assert ctx.constraints["limited_width"] is True
    assert ctx.constraints["limited_height"] is True
    assert ctx.optimization_paths["default_output_mode"] == "ascii"


def test_constraints_high_perf():
    ctx_mod = load_module("glyph_stream_context_perf")
    ctx = ctx_mod.SystemContext(check_network=False)
    ctx.attributes["cpu_count"] = 16
    ctx.attributes["memory_available"] = 32 * 1024 * 1024 * 1024
    ctx.attributes["has_cuda"] = True
    assert ctx._estimate_performance_tier() >= 2


def test_refresh(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_refresh")
    monkeypatch.setattr(
        ctx_mod.SystemContext, "_check_network_connection", lambda self: False
    )
    ctx = ctx_mod.SystemContext(check_network=False)
    ctx.refresh(check_network=True)
    assert "optimization_level" in ctx.get_environment_summary()


def test_color_support_term(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_term")
    ctx = ctx_mod.SystemContext(check_network=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert ctx._detect_color_support({"platform": "Linux"}) is False
    monkeypatch.setenv("TERM", "xterm-256color")
    assert ctx._detect_color_support({"platform": "Linux"}) is True
    monkeypatch.delenv("TERM", raising=False)


def test_hardware_accel_torch_import_error(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_torch_err")
    monkeypatch.setenv("EIDOSIAN_ENABLE_TORCH", "1")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ctx = ctx_mod.SystemContext(check_network=False)
    assert ctx.attributes.get("has_cuda") is False


def test_network_check_connection_paths(monkeypatch):
    ctx_mod = load_module("glyph_stream_context_socket")
    import socket as socket_module

    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyError(Exception):
        pass

    class DummyTimeout(Exception):
        pass

    def ok_conn(*_args, **_kwargs):
        return DummySocket()

    monkeypatch.setattr(socket_module, "create_connection", ok_conn)
    ctx = ctx_mod.SystemContext(check_network=False)
    assert ctx._check_network_connection() is True

    def bad_conn(*_args, **_kwargs):
        raise DummyError("nope")

    monkeypatch.setattr(socket_module, "error", DummyError)
    monkeypatch.setattr(socket_module, "timeout", DummyTimeout)
    monkeypatch.setattr(socket_module, "create_connection", bad_conn)
    assert ctx._check_network_connection() is False


def test_constraints_medium_perf():
    ctx_mod = load_module("glyph_stream_context_medium")
    ctx = ctx_mod.SystemContext(check_network=False)
    ctx.attributes["cpu_count"] = 4
    ctx.attributes["memory_available"] = 4 * 1024 * 1024 * 1024
    ctx.attributes["has_cuda"] = False
    constraints = ctx._detect_constraints()
    assert constraints["default_fps"] == 10


def test_optimization_paths_output_modes():
    ctx_mod = load_module("glyph_stream_context_paths")
    ctx = ctx_mod.SystemContext(check_network=False)
    ctx.attributes["cpu_physical"] = 8
    ctx.constraints = {
        "limited_width": False,
        "performance_tier": 2,
        "max_scale_factor": 4,
    }

    ctx.capabilities = {
        "can_display_color": True,
        "can_display_unicode": False,
        "has_high_performance": True,
    }
    paths = ctx._generate_optimization_paths()
    assert paths["block_width"] == 6
    assert paths["default_output_mode"] == "ascii-color"

    ctx.capabilities = {
        "can_display_color": True,
        "can_display_unicode": True,
        "has_high_performance": True,
    }
    paths = ctx._generate_optimization_paths()
    assert paths["default_output_mode"] == "unicode-color"


def test_get_optimized_parameters():
    ctx_mod = load_module("glyph_stream_context_params")
    ctx = ctx_mod.SystemContext(check_network=False)
    params = ctx.get_optimized_parameters()
    assert "scale_factor" in params
