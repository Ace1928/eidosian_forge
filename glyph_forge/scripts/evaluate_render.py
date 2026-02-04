#!/usr/bin/env python3
"""Evaluate glyph render quality against source images."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

from glyph_forge.streaming.core.renderer import GlyphRenderer, RenderConfig
from glyph_forge.streaming.core.recorder import GlyphRecorder, RecorderConfig


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(m))


def evaluate_image(path: Path, width: int = 80, height: int = 40) -> None:
    src = cv2.imread(str(path))
    if src is None:
        print(f"Failed to read {path}")
        return
    src = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)

    renderer = GlyphRenderer(RenderConfig(color="ansi256", dithering=True, gamma=1.1))
    glyph = renderer.render(src, width=width, height=height, color="ansi256")

    recorder = GlyphRecorder(RecorderConfig(output_path=Path("/tmp/glyph_eval.mp4")))
    rendered = recorder.render_to_image(glyph)
    rendered = cv2.resize(rendered, (width, height), interpolation=cv2.INTER_AREA)

    print(f"{path.name}: MSE={mse(src, rendered):.2f}, PSNR={psnr(src, rendered):.2f} dB")


def main() -> None:
    samples = [
        Path("assets/demo.png"),
    ]
    for sample in samples:
        if sample.exists():
            evaluate_image(sample, width=120, height=60)


if __name__ == "__main__":
    main()
