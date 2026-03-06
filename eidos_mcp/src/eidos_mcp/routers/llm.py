from __future__ import annotations

import json
from typing import Optional

from eidosian_core import eidosian
from llm_forge.benchmarking.engine_bench import Benchmarker
from llm_forge.engine.local_cli import EngineConfig, LocalCLIEngine

from ..config.models import get_model_config
from ..core import tool

DEFAULT_MODEL = get_model_config().local_model_path


@tool(
    name="llm_local_generate",
    description="Generate text using a local GGUF model via llama-cli.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "system_prompt": {"type": "string"},
            "model_path": {"type": "string", "default": DEFAULT_MODEL},
            "temp": {"type": "number", "default": 0.7},
            "n_predict": {"type": "integer", "default": 1024},
        },
        "required": ["prompt"],
    },
)
@eidosian()
async def llm_local_generate(
    prompt: str, system_prompt: Optional[str] = None, model_path: str = DEFAULT_MODEL, **kwargs
) -> str:
    """Generate text locally."""
    config = EngineConfig(model_path=model_path, temp=kwargs.get("temp", 0.7))
    engine = LocalCLIEngine(config)
    return await engine.generate(prompt, system_prompt=system_prompt, n_predict=kwargs.get("n_predict", 1024))


@tool(
    name="llm_run_benchmark",
    description="Run a performance benchmark on a specific GGUF model.",
    parameters={"type": "object", "properties": {"model_path": {"type": "string", "default": DEFAULT_MODEL}}},
)
@eidosian()
async def llm_run_benchmark(model_path: str = DEFAULT_MODEL) -> str:
    """Run benchmark."""
    bench = Benchmarker()
    results = await bench.run_throughput_test(model_path)
    if not results:
        return "Benchmark failed or binary missing."
    return json.dumps([r.model_dump() for r in results], indent=2)
