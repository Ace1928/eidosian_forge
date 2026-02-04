"""
Benchmark Script.
"""
import time
import sys
import os
import cProfile
import pstats
import numpy as np

# Path adjustment
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pyparticles.core.types import SimulationConfig
from pyparticles.physics.engine import PhysicsEngine

def run_benchmark():
    N = 10000
    print(f"Benchmarking Physics Engine (N={N})...")
    
    cfg = SimulationConfig.default()
    cfg.num_particles = N
    engine = PhysicsEngine(cfg)
    
    # Warmup
    print("Warming up JIT...")
    engine.update(0.01)
    
    print("Starting Profile Run (100 frames)...")
    t0 = time.time()
    for _ in range(100):
        engine.update(0.02)
    t1 = time.time()
    
    dt = t1 - t0
    fps = 100 / dt
    print(f"Done. Time: {dt:.4f}s, FPS: {fps:.2f}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_benchmark()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
