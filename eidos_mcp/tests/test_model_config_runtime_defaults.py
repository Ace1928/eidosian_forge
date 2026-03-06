from __future__ import annotations

from pathlib import Path

from eidos_mcp.config.models import get_model_config
from llm_forge.benchmarking.engine_bench import Benchmarker
from llm_forge.engine.base import EngineConfig
from llm_forge.engine.local_cli import LocalCLIEngine


def test_runtime_model_defaults_are_centralized() -> None:
    cfg = get_model_config()

    assert cfg.inference_model == "qwen3.5:2b"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.inference.thinking_mode == "off"
    assert cfg.local_model_path.endswith("Qwen2.5-1.5B-Instruct-Q8_0.gguf")
    assert cfg.local_inference.llama_cli_path.endswith("llama.cpp/build/bin/llama-cli")
    assert cfg.local_inference.llama_bench_path.endswith("llama.cpp/build/bin/llama-bench")


def test_local_llama_cpp_wrappers_use_centralized_paths() -> None:
    cfg = get_model_config()
    engine = LocalCLIEngine(EngineConfig(model_path=cfg.local_model_path))
    bench = Benchmarker()

    assert Path(engine.bin_path).as_posix().endswith("llama.cpp/build/bin/llama-cli")
    assert Path(bench.bin_path).as_posix().endswith("llama.cpp/build/bin/llama-bench")
