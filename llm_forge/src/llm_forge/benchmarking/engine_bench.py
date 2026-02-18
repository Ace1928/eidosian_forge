from __future__ import annotations
import subprocess
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from eidosian_core import eidosian

class BenchResult(BaseModel):
    """Parsed results from llama-bench."""
    model: str
    n_prompt: int
    n_gen: int
    t_prompt_ms: float
    t_gen_ms: float
    tokens_sec_prompt: float
    tokens_sec_gen: float

class Benchmarker:
    """Wrapper for llama-bench to evaluate local engine performance."""

    def __init__(self, bin_path: Optional[Path] = None):
        self.bin_path = bin_path or Path(__file__).resolve().parents[4] / "llm_forge/bin/llama-bench"

    @eidosian()
    async def run_throughput_test(self, model_path: str, p_tokens: List[int] = [512], g_tokens: List[int] = [128]) -> List[BenchResult]:
        """Run standard throughput benchmark."""
        if not self.bin_path.exists():
            return []

        # Convert lists to comma-separated strings for llama-bench
        p_str = ",".join(map(str, p_tokens))
        g_str = ",".join(map(str, g_tokens))

        cmd = [
            str(self.bin_path),
            "-m", model_path,
            "-p", p_str,
            "-n", g_str,
            "-o", "json" # Output as JSON for easy parsing
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode != 0:
                return []

            raw_results = json.loads(stdout.decode())
            parsed = []
            for r in raw_results:
                n_prompt = r.get("n_prompt", 0)
                n_gen = r.get("n_gen", 0)
                avg_ts = r.get("avg_ts", 0.0)
                
                parsed.append(BenchResult(
                    model=r.get("model_type", ""),
                    n_prompt=n_prompt,
                    n_gen=n_gen,
                    t_prompt_ms=0.0, # Not provided directly in summary
                    t_gen_ms=0.0,    # Not provided directly in summary
                    tokens_sec_prompt=avg_ts if n_prompt > 0 and n_gen == 0 else 0.0,
                    tokens_sec_gen=avg_ts if n_gen > 0 else 0.0
                ))
            return parsed
        except Exception:
            return []
