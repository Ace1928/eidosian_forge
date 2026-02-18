from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Optional
from eidosian_core import eidosian

class Quantizer:
    """Wrapper for llama-quantize to optimize model weights."""

    def __init__(self, bin_path: Optional[Path] = None):
        self.bin_path = bin_path or Path(__file__).resolve().parents[4] / "llm_forge/bin/llama-quantize"

    @eidosian()
    async def quantize(self, input_path: str, output_path: str, method: str = "Q4_K_M") -> bool:
        """Quantize a GGUF model."""
        if not self.bin_path.exists():
            return False

        cmd = [
            str(self.bin_path),
            input_path,
            output_path,
            method
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False
