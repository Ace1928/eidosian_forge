from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import Generator, Optional
from .base import BaseEngine, EngineConfig
from eidosian_core import eidosian

class LocalCLIEngine(BaseEngine):
    """Engine using llama-completion for deterministic single-shot inference."""

    def __init__(self, config: EngineConfig, bin_path: Optional[Path] = None):
        super().__init__(config)
        self.bin_path = bin_path or Path(__file__).resolve().parents[4] / "llm_forge/bin/llama-completion"

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        bin_dir = str(self.bin_path.parent.resolve())
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        ld_library = env.get("LD_LIBRARY_PATH", "")
        if ld_library:
            if f":{bin_dir}:" not in f":{ld_library}:":
                env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_library}"
        else:
            env["LD_LIBRARY_PATH"] = bin_dir
        return env

    @eidosian()
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Call llama-cli and return the result."""
        if not self.bin_path.exists():
            return f"ERROR: llama-cli not found at {self.bin_path}. Build may be in progress."

        # Construct ChatML or Qwen style prompt if system prompt provided
        if system_prompt:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        else:
            full_prompt = prompt

        cmd = [
            str(self.bin_path),
            "-m", self.config.model_path,
            "-p", full_prompt,
            "-n", str(kwargs.get("n_predict", self.config.n_predict)),
            "-c", str(kwargs.get("ctx_size", self.config.ctx_size)),
            "-t", str(kwargs.get("threads", self.config.threads)),
            "--temp", str(kwargs.get("temp", self.config.temp)),
            "--simple-io",
            "--no-display-prompt",
            "--no-warmup",
            "-no-cnv",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return f"ERROR: Engine failed ({proc.returncode}): {stderr.decode()}"
            return stdout.decode().strip()
        except Exception as e:
            return f"ERROR: Subprocess exception: {e}"

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        # CLI streaming is complex via subprocess, usually better via llama-server
        raise NotImplementedError("Streaming not supported in LocalCLIEngine. Use ServerEngine.")
