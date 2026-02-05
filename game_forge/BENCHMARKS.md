# Benchmarks: game_forge

**Date**: 2026-02-05

## Stratum
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/src/Stratum/tests/benchmark.py --grid 16 --ticks 10 --output game_forge/src/Stratum/tests/artifacts/benchmark_single_20260205.json`
- Results (16x16, 10 ticks): elapsed_seconds=29.871, ticks_per_second=0.335, microticks_per_second=15.868, cells_per_second=85.702, peak_memory_mb=0.444
- Note: scaling benchmark with multiple grid sizes was stopped due to long runtime.

## falling_sand
- Command: `PYTHONPATH=src /home/lloyd/eidosian_forge/eidosian_venv/bin/python -m falling_sand.benchmarks --runs 5 --output artifacts/benchmark_20260205.json`
- Results (mean seconds over 5 runs): indexer=0.052, simulation=0.004, streaming=0.705, terrain=0.014
- Report: `game_forge/tools/artifacts/falling_sand_benchmark_20260205.json`

## gene_particles
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/tools/gene_particles_benchmark.py --steps 200 --cell-types 4 --particles 200 --output game_forge/tools/artifacts/gene_particles_benchmark_20260205.json`
- Results: total_s=5.817, mean_ms=29.086, median_ms=24.619, p95_ms=33.680, p99_ms=101.155, steps_per_s=34.380
- Warning: pygame reported missing system font 'Fira Sans' (headless defaulted to bundled font).

## algorithms_lab
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/tools/algorithms_lab/benchmark.py --algorithms all --particles 256 --steps 50 --output game_forge/tools/artifacts/algorithms_lab_benchmark_20260205.json`
- Results (seconds): grid=0.162, neighbor-list=0.001, barnes-hut=0.036, fmm2d=0.797, fmm-ml=1.734, forces=0.030, sph=0.025, pbf=0.055, xpbd=0.055
- Warning: numba could not obtain multiprocessing lock due to /dev/shm permissions.

## pyparticles
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/pyparticles/benchmarks/benchmark.py --particles 512 --steps 10 --dt 0.01 --warmup 1 --no-profile --output game_forge/pyparticles/benchmarks/artifacts/benchmark_engine_20260205.json`
- Results: elapsed_seconds=0.757, fps=13.203, particles=512, steps=10, dt=0.01
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/pyparticles/benchmarks/benchmark_sim.py --particles 512 --steps 10 --dt 0.01 --warmup 1 --no-profile --output game_forge/pyparticles/benchmarks/artifacts/benchmark_sim_20260205.json`
- Results: elapsed_seconds=0.081, fps=123.672, particles=512, steps=10, dt=0.01
- Warning: numba could not obtain multiprocessing lock due to /dev/shm permissions.
