from __future__ import annotations
import os
import subprocess
import tempfile
import asyncio
from pathlib import Path
from typing import Generator, List, Optional
from .base import BaseEngine, EngineConfig
from eidosian_core import eidosian

class LocalCLIEngine(BaseEngine):
    """Engine using the llama-cli binary for inference."""

    def __init__(self, config: EngineConfig, bin_path: Optional[Path] = None):
        super().__init__(config)
        self.bin_path = bin_path or Path(__file__).resolve().parents[4] / "llm_forge/bin/llama-cli"

    @eidosian()
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Call llama-cli and return the result."""
        if not self.bin_path.exists():
            return f"ERROR: llama-cli not found at {self.bin_path}. Build may be in progress."

        # Construct ChatML or Qwen style prompt if system prompt provided
        if system_prompt:
            full_prompt = f"<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"
        else:
            full_prompt = prompt

        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as tmp:
            tmp.write(full_prompt)
            tmp_path = tmp.name

        cmd = [
            str(self.bin_path),
            "-m", self.config.model_path,
            "-f", tmp_path,
            "-n", str(kwargs.get("n_predict", self.config.n_predict)),
            "-c", str(kwargs.get("ctx_size", self.config.ctx_size)),
            "-t", str(kwargs.get("threads", self.config.threads)),
            "--temp", str(kwargs.get("temp", self.config.temp)),
            "--simple-io",
            "-st", # Single turn
            "-e"   # Escape processing
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            os.unlink(tmp_path)

            if proc.returncode != 0:
                return f"ERROR: Engine failed ({proc.returncode}): {stderr.decode()}"
            
            return stdout.decode().strip()
        except Exception as e:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            return f"ERROR: Subprocess exception: {e}"

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        # CLI streaming is complex via subprocess, usually better via llama-server
        raise NotImplementedError("Streaming not supported in LocalCLIEngine. Use ServerEngine.")
