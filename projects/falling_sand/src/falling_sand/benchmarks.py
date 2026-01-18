"""Benchmark suite for engine and tooling."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from falling_sand.engine.config import VoxelConfig
from falling_sand.engine.materials import Material
from falling_sand.engine.simulation import step_world
from falling_sand.engine.streaming import ChunkStreamer, StreamConfig
from falling_sand.engine.terrain import TerrainConfig, TerrainGenerator
from falling_sand.engine.world import World
from falling_sand.indexer import index_project


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the benchmark suite."""

    parser = argparse.ArgumentParser(description="Run benchmark suite.")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    return parser


def _summarize_samples(name: str, samples: list[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("samples must be non-empty")
    return {
        "name": name,
        "runs": len(samples),
        "samples": samples,
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "stdev_seconds": statistics.pstdev(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def benchmark_indexer(source_root: Path, tests_root: Path, runs: int) -> list[float]:
    """Benchmark the indexer and return run durations in seconds."""

    if runs <= 0:
        raise ValueError("runs must be positive")

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        index_project(source_root, tests_root)
        samples.append(time.perf_counter() - start)
    return samples


def benchmark_simulation(runs: int, size: int = 16) -> list[float]:
    """Benchmark a simple 3D simulation step on one chunk."""

    if runs <= 0:
        raise ValueError("runs must be positive")

    config = VoxelConfig(voxel_size_m=0.1, chunk_size_voxels=size)
    world = World(config=config)
    chunk = world.ensure_chunk((0, 0, 0))
    data = np.random.randint(0, 5, size=(size, size, size), dtype=np.uint8)
    data[data == int(Material.SOLID)] = int(Material.GRANULAR)
    chunk.data = data

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        step_world(world.chunks)
        samples.append(time.perf_counter() - start)
    return samples


def benchmark_streaming(runs: int, radius: int = 1) -> list[float]:
    """Benchmark chunk streaming updates."""

    if runs <= 0:
        raise ValueError("runs must be positive")

    config = VoxelConfig()
    world = World(config=config)
    streamer = ChunkStreamer(world, StreamConfig(radius=radius))
    samples: list[float] = []
    focus = (0, 0, 0)
    for step in range(runs):
        start = time.perf_counter()
        focus = (focus[0] + 1, focus[1] + (step % 2), focus[2])
        streamer.update_focus(focus)
        samples.append(time.perf_counter() - start)
    return samples


def benchmark_terrain_generation(runs: int, size: int = 10) -> list[float]:
    """Benchmark terrain chunk generation."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    if size <= 0:
        raise ValueError("size must be positive")

    config = TerrainConfig()
    generator = TerrainGenerator(config)
    samples: list[float] = []
    for step in range(runs):
        coord = (step % 4, (step // 2) % 4, 0)
        start = time.perf_counter()
        generator.chunk(coord, size)
        samples.append(time.perf_counter() - start)
    return samples


def write_benchmark_report(benchmarks: list[dict[str, object]], output: Path) -> None:
    """Write benchmark stats to a JSON report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"benchmarks": benchmarks}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark suite CLI."""

    args = build_parser().parse_args(argv)
    benchmarks = [
        _summarize_samples("indexer", benchmark_indexer(args.source_root, args.tests_root, args.runs)),
        _summarize_samples("simulation", benchmark_simulation(args.runs)),
        _summarize_samples("streaming", benchmark_streaming(args.runs)),
        _summarize_samples("terrain", benchmark_terrain_generation(args.runs)),
    ]
    write_benchmark_report(benchmarks, args.output)
    return 0
