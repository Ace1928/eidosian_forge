from __future__ import annotations

from pathlib import Path

from eidos_mcp.config.models import get_model_config
from eidosian_core.ports import get_service_url
from llm_forge.benchmarking.engine_bench import Benchmarker
from llm_forge.engine.base import EngineConfig
from llm_forge.engine.local_cli import LocalCLIEngine


def test_runtime_model_defaults_are_centralized() -> None:
    cfg = get_model_config()

    assert cfg.inference_model == "qwen3.5:2b"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.inference.thinking_mode == "off"
    assert cfg.ollama.base_url == get_service_url(
        "ollama_qwen_http", default_port=8938, default_host="127.0.0.1", default_path=""
    ).rstrip("/")
    assert cfg.ollama.embedding_base_url == get_service_url(
        "ollama_embedding_http", default_port=8940, default_host="127.0.0.1", default_path=""
    ).rstrip("/")
    assert (
        cfg.local_model_path.endswith("Qwen3.5-2B-Instruct-Q4_K_M.gguf")
        or cfg.local_model_path.endswith("Qwen3.5-2B-Instruct-Q8_0.gguf")
        or cfg.local_model_path.endswith("Qwen3.5-2B-Instruct-Q8_0.gguf")
    )
    assert cfg.local_inference.llama_cli_path.endswith("llama.cpp/build/bin/llama-cli")
    assert cfg.local_inference.llama_bench_path.endswith("llama.cpp/build/bin/llama-bench")


def test_local_llama_cpp_wrappers_use_centralized_paths() -> None:
    cfg = get_model_config()
    engine = LocalCLIEngine(EngineConfig(model_path=cfg.local_model_path))
    bench = Benchmarker()

    assert Path(engine.bin_path).as_posix().endswith("llama.cpp/build/bin/llama-cli")
    assert Path(bench.bin_path).as_posix().endswith("llama.cpp/build/bin/llama-bench")
