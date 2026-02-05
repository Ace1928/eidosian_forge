#!/usr/bin/env python3
"""
Performance benchmarks for Stratum.

This script measures simulation performance at various grid sizes
and provides baseline metrics for tracking performance regressions.

Usage:
    python3 tests/benchmark.py
    python3 tests/benchmark.py --grid 256 --ticks 100
    python3 tests/benchmark.py --profile
"""

import argparse
import json
import time
import sys
import os
import tempfile
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np
from eidosian_core import eidosian

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EngineConfig, DeterminismMode
from core.fabric import Fabric
from core.ledger import Ledger
from core.quanta import Quanta
from core.registry import SpeciesRegistry
from domains.materials.fundamentals import MaterialsFundamentals


@eidosian()
def benchmark_simulation(
    grid_size: int,
    num_ticks: int,
    microticks_per_tick: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """Run a benchmark simulation and collect metrics.
    
    Args:
        grid_size: Grid width and height.
        num_ticks: Number of ticks to run.
        microticks_per_tick: Microtick budget per tick.
        seed: Random seed.
        
    Returns:
        Dictionary with benchmark metrics.
    """
    he_props = [
        "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
        "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
    ]
    
    cfg = EngineConfig(
        grid_w=grid_size,
        grid_h=grid_size,
        base_seed=seed,
        entropy_mode=False,
        determinism_mode=DeterminismMode.REPLAY_DETERMINISTIC,
        microtick_cap_per_region=5,
        active_region_max=min(grid_size * grid_size // 4, 2048),
        viscosity_global=0.05,
        thermal_pressure_coeff=0.05,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = SpeciesRegistry(registry_path, he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize
        stellar_id = materials.stellar_species.id
        rng = np.random.default_rng(seed)
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                mass = 0.5 + 0.5 * rng.random()
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 0.5 + 0.25 * rng.random()
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
        
        # Measure initialization memory
        import tracemalloc
        tracemalloc.start()
        
        # Run simulation
        start_time = time.perf_counter()
        total_microticks = 0
        
        for tick in range(1, num_ticks + 1):
            quanta.step(tick, microticks_per_tick)
            total_microticks = quanta.total_microticks
        
        end_time = time.perf_counter()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed = end_time - start_time
        cells = grid_size * grid_size
        
        return {
            'grid_size': grid_size,
            'num_ticks': num_ticks,
            'total_cells': cells,
            'total_microticks': total_microticks,
            'elapsed_seconds': elapsed,
            'ticks_per_second': num_ticks / elapsed,
            'microticks_per_second': total_microticks / elapsed,
            'cells_per_second': cells * num_ticks / elapsed,
            'peak_memory_mb': peak / 1024 / 1024,
            'current_memory_mb': current / 1024 / 1024,
        }


@eidosian()
def run_scaling_benchmark(
    grid_sizes: List[int] = [16, 32, 64],
    num_ticks: int = 20
) -> List[Dict[str, float]]:
    """Run benchmarks at multiple grid sizes.
    
    Args:
        grid_sizes: List of grid sizes to benchmark.
        num_ticks: Number of ticks per benchmark.
        
    Returns:
        List of benchmark results.
    """
    results = []
    
    for size in grid_sizes:
        print(f"Benchmarking {size}x{size} grid...")
        result = benchmark_simulation(size, num_ticks)
        results.append(result)
        print(f"  {result['ticks_per_second']:.2f} ticks/sec, "
              f"{result['microticks_per_second']:.0f} microticks/sec, "
              f"{result['peak_memory_mb']:.1f} MB peak")
    
    return results


@eidosian()
def profile_simulation(grid_size: int = 64, num_ticks: int = 20):
    """Profile the simulation to identify hotspots.
    
    Args:
        grid_size: Grid size for profiling.
        num_ticks: Number of ticks to profile.
    """
    import cProfile
    import pstats
    import io
    
    he_props = [
        "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
        "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
    ]
    
    cfg = EngineConfig(
        grid_w=grid_size,
        grid_h=grid_size,
        base_seed=42,
        microtick_cap_per_region=5,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = SpeciesRegistry(registry_path, he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                mass = 1.0
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 0.5
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
        
        # Profile
        pr = cProfile.Profile()
        pr.enable()
        
        for tick in range(1, num_ticks + 1):
            quanta.step(tick, 100)
        
        pr.disable()
        
        # Print results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)
        print(s.getvalue())


def format_single_payload(result: Dict[str, float], ticks: int) -> Dict[str, Any]:
    return {
        "mode": "single",
        "grid_size": int(result.get("grid_size", 0)),
        "ticks": ticks,
        "result": result,
    }


def format_scaling_payload(results: List[Dict[str, float]], ticks: int) -> Dict[str, Any]:
    return {
        "mode": "scaling",
        "ticks": ticks,
        "results": results,
    }


def format_profile_payload(grid_size: int, ticks: int) -> Dict[str, Any]:
    return {
        "mode": "profile",
        "grid_size": grid_size,
        "ticks": ticks,
        "note": "profile output is printed to stdout",
    }


def write_payload(path: str, payload: Dict[str, Any]) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stratum performance benchmarks")
    parser.add_argument('--grid', type=int, default=0, help='Single grid size to benchmark (0 for scaling test)')
    parser.add_argument('--ticks', type=int, default=50, help='Number of ticks')
    parser.add_argument('--profile', action='store_true', help='Run profiler instead of benchmark')
    parser.add_argument('--output', default="", help='Write JSON benchmark output')
    args = parser.parse_args(argv)
    
    print("=" * 60)
    print("Stratum Performance Benchmark")
    print("=" * 60)
    print()
    
    if args.profile:
        print("Profiling simulation...")
        profile_simulation(grid_size=args.grid or 64, num_ticks=args.ticks)
        if args.output:
            write_payload(args.output, format_profile_payload(args.grid or 64, args.ticks))
    elif args.grid > 0:
        # Single benchmark
        result = benchmark_simulation(args.grid, args.ticks)
        print(f"\nResults for {args.grid}x{args.grid} grid, {args.ticks} ticks:")
        print(f"  Total time: {result['elapsed_seconds']:.3f} seconds")
        print(f"  Ticks/second: {result['ticks_per_second']:.2f}")
        print(f"  Microticks/second: {result['microticks_per_second']:.0f}")
        print(f"  Cells processed/second: {result['cells_per_second']:.0f}")
        print(f"  Peak memory: {result['peak_memory_mb']:.1f} MB")
        if args.output:
            write_payload(args.output, format_single_payload(result, args.ticks))
    else:
        # Scaling benchmark
        print("Running scaling benchmark...")
        print()
        results = run_scaling_benchmark(num_ticks=args.ticks)
        
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print()
        print(f"{'Grid':>10} {'Ticks/s':>12} {'μticks/s':>12} {'Memory (MB)':>12}")
        print("-" * 50)
        for r in results:
            print(f"{r['grid_size']:>10} {r['ticks_per_second']:>12.2f} "
                  f"{r['microticks_per_second']:>12.0f} {r['peak_memory_mb']:>12.1f}")
        if args.output:
            write_payload(args.output, format_scaling_payload(results, args.ticks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
