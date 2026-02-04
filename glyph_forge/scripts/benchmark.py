#!/usr/bin/env python3
"""
⚡ Glyph Forge Benchmark ⚡

Measure the raw performance of the Unified Stream Engine.
"""
import sys
import os
import time
import numpy as np
from statistics import mean, stdev

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from glyph_forge.streaming.engine import UnifiedStreamEngine, UnifiedStreamConfig
from glyph_forge.streaming.core.renderer import GlyphRenderer, RenderConfig

def benchmark_renderer(resolution=(1280, 720), iterations=100):
    """Benchmark raw rendering performance."""
    width, height = resolution
    # Terminal size approx (chars)
    term_w, term_h = 160, 45
    
    print(f"Benchmarking Renderer: Input {width}x{height} -> Output {term_w}x{term_h}")
    
    config = RenderConfig(mode="gradient", color="ansi256")
    renderer = GlyphRenderer(config)
    
    # Pre-generate frames to avoid allocation overhead in loop
    frames = [
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    times = []
    print(f"Running {iterations} iterations...")
    
    start_total = time.perf_counter()
    for i in range(iterations):
        frame = frames[i % 10]
        t0 = time.perf_counter()
        renderer.render(frame, width=term_w, height=term_h)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000) # ms
    
    total_time = time.perf_counter() - start_total
    
    avg_ms = mean(times)
    std_ms = stdev(times)
    fps = 1000 / avg_ms
    
    print(f"Results:")
    print(f"  Avg Time: {avg_ms:.2f} ms ± {std_ms:.2f}")
    print(f"  Est FPS:  {fps:.2f}")
    print(f"  Total:    {total_time:.2f}s")
    
    return fps

def main():
    print("⚡ Eidosian Benchmark ⚡")
    print("=" * 30)
    
    resolutions = [
        ("480p", (854, 480)),
        ("720p", (1280, 720)),
        ("1080p", (1920, 1080))
    ]
    
    for name, res in resolutions:
        print(f"\n[{name}]")
        benchmark_renderer(res)

if __name__ == "__main__":
    main()
