import sys
import os
import time
import cProfile
import pstats
import numba

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pyparticles.simulation import ParticleSystem
from pyparticles.config import SimConfig

def benchmark():
    N = 10000
    print(f"Benchmarking with N={N} particles...")
    cfg = SimConfig.default()
    cfg.num_particles = N
    
    sim = ParticleSystem(cfg)
    
    # Warmup
    print("Warming up JIT...")
    sim.update(0.02)
    print("Warmup complete.")
    
    start_time = time.time()
    steps = 100
    for _ in range(steps):
        sim.update(0.02)
    end_time = time.time()
    
    duration = end_time - start_time
    fps = steps / duration
    print(f"Processed {steps} frames in {duration:.4f}s")
    print(f"Simulation FPS: {fps:.2f}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    benchmark()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)