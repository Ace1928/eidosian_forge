# Benchmarks: game_forge

**Date**: 2026-02-05

## Benchmark Suite (quick)
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/tools/benchmark_suite.py --preset quick --output-dir game_forge/tools/artifacts --tag 20260205_quick`
- Results:
  - gene_particles: total_s=5.192, mean_ms=51.920, median_ms=47.437, p95_ms=98.672, steps_per_s=19.260
  - algorithms_lab (seconds): grid=0.140, neighbor-list=0.001, barnes-hut=0.060, fmm2d=0.792, fmm-ml=1.756, forces=0.030, sph=0.026, pbf=0.055, xpbd=0.056
  - falling_sand (mean seconds over 3 runs): indexer=0.051, simulation=0.004, streaming=0.670, terrain=0.015
  - stratum (16x16, 5 ticks): elapsed_seconds=17.849, ticks_per_second=0.280, microticks_per_second=16.192, cells_per_second=71.714, peak_memory_mb=0.358
  - pyparticles engine: elapsed_seconds=0.019, fps=264.186, particles=256, steps=5, dt=0.01
  - pyparticles sim: elapsed_seconds=0.016, fps=321.630, particles=256, steps=5, dt=0.01
- Skipped: agentic-chess (missing python-chess module)
- Warnings: pygame system font 'Fira Sans' missing; numba could not obtain multiprocessing lock due to /dev/shm permissions.

## Benchmark Suite (full)
- Command: `/home/lloyd/eidosian_forge/eidosian_venv/bin/python game_forge/tools/benchmark_suite.py --preset full --output-dir game_forge/tools/artifacts --tag 20260205_full`
- Results:
  - gene_particles: total_s=28.831, mean_ms=57.661, median_ms=57.379, p95_ms=60.942, steps_per_s=17.343
  - algorithms_lab (seconds): grid=0.155, neighbor-list=0.002, barnes-hut=0.103, fmm2d=2.058, fmm-ml=5.059, forces=0.133, sph=0.093, pbf=0.218, xpbd=0.219
  - falling_sand (mean seconds over 10 runs): indexer=0.053, simulation=0.003, streaming=0.815, terrain=0.015
  - stratum (32x32, 20 ticks): elapsed_seconds=39.460, ticks_per_second=0.507, microticks_per_second=57.146, cells_per_second=519.002, peak_memory_mb=1.123
  - pyparticles engine: elapsed_seconds=0.078, fps=129.026, particles=512, steps=10, dt=0.01
  - pyparticles sim: elapsed_seconds=0.084, fps=119.546, particles=512, steps=10, dt=0.01
- Skipped: agentic-chess (missing python-chess module)
- Warnings: pygame system font 'Fira Sans' missing; numba could not obtain multiprocessing lock due to /dev/shm permissions.

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
