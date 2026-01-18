#!/usr/bin/env python3
"""
Golden Checksum Harness for Stratum.

This script produces verifiable checksums for determinism testing.
Run twice to verify identical results.

Usage:
    python3 -m tests.golden_checksum --seed 42 --ticks 100
    
Output format:
    {
        "engine_version": "git SHA or version",
        "world_law_version": "1.0.0",
        "config_hash": "sha256 of config",
        "seed": 42,
        "num_ticks": 100,
        "field_checksums": {
            "rho": "sha256",
            "px": "sha256",
            ...
        },
        "registry_checksum": "sha256",
        "final_state_checksum": "sha256 of everything"
    }
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EngineConfig, DeterminismMode
from core.fabric import Fabric
from core.ledger import Ledger
from core.quanta import Quanta
from core.registry import SpeciesRegistry
from domains.materials.fundamentals import MaterialsFundamentals


def get_git_sha() -> str:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def hash_array(arr: np.ndarray) -> str:
    """Compute SHA256 hash of a numpy array."""
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


def hash_config(cfg: EngineConfig) -> str:
    """Compute SHA256 hash of configuration."""
    h = hashlib.sha256()
    config_dict = cfg.to_dict()
    # Sort keys for deterministic ordering
    for k in sorted(config_dict.keys()):
        v = config_dict[k]
        h.update(f"{k}={v}".encode('utf-8'))
    return h.hexdigest()


def hash_registry(registry: SpeciesRegistry) -> str:
    """Compute SHA256 hash of species registry."""
    h = hashlib.sha256()
    for sid in sorted(registry.species.keys()):
        sp = registry.species[sid]
        h.update(f"species:{sid}:".encode('utf-8'))
        for prop in sorted(sp.he_props.keys()):
            h.update(f"{prop}={sp.he_props[prop]:.10f}".encode('utf-8'))
    return h.hexdigest()


def hash_mixtures(fabric: Fabric) -> str:
    """Compute SHA256 hash of mixture data."""
    h = hashlib.sha256()
    for i in range(fabric.cfg.grid_w):
        for j in range(fabric.cfg.grid_h):
            mix = fabric.mixtures[i][j]
            h.update(f"mix_{i}_{j}:".encode('utf-8'))
            for sid, mass in zip(mix.species_ids, mix.masses):
                h.update(f"{sid}:{mass:.15e}".encode('utf-8'))
    return h.hexdigest()


def run_golden_checksum(seed: int, num_ticks: int, verbose: bool = False) -> dict:
    """Run simulation and compute golden checksums.
    
    Args:
        seed: Base seed for simulation.
        num_ticks: Number of ticks to run.
        verbose: If True, print progress.
        
    Returns:
        Dictionary with all checksums and metadata.
    """
    he_props = [
        "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
        "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
    ]
    
    # Create config with strict determinism
    cfg = EngineConfig(
        grid_w=16, grid_h=16,
        base_seed=seed,
        entropy_mode=False,
        determinism_mode=DeterminismMode.STRICT_DETERMINISTIC,
        microtick_cap_per_region=3,
        active_region_max=64,
        viscosity_global=0.05,
        thermal_pressure_coeff=0.05,
    )
    
    # Create temporary registry file
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry.json")
        registry = SpeciesRegistry(registry_path, he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with deterministic values
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                # Deterministic initialization based on position and seed
                mass = 0.5 + 0.5 * ((i + j + seed) % 100) / 100.0
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 0.5 + 0.25 * ((i * j + seed) % 50) / 50.0
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
        
        # Record initial checksums
        initial_checksums = {
            'rho': hash_array(fabric.rho),
            'px': hash_array(fabric.px),
            'py': hash_array(fabric.py),
            'E_heat': hash_array(fabric.E_heat),
            'E_rad': hash_array(fabric.E_rad),
        }
        
        # Run simulation
        for tick in range(1, num_ticks + 1):
            quanta.step(tick, 100)
            if verbose and tick % (num_ticks // 10 or 1) == 0:
                print(f"  Tick {tick}/{num_ticks}, mass={fabric.rho.sum():.6f}")
        
        # Compute final checksums
        field_checksums = {
            'rho': hash_array(fabric.rho),
            'px': hash_array(fabric.px),
            'py': hash_array(fabric.py),
            'E_heat': hash_array(fabric.E_heat),
            'E_rad': hash_array(fabric.E_rad),
            'influence': hash_array(fabric.influence),
            'BH_mass': hash_array(fabric.BH_mass),
            'EH_mask': hash_array(fabric.EH_mask),
        }
        
        mixture_checksum = hash_mixtures(fabric)
        registry_checksum = hash_registry(registry)
        
        # Compute final state checksum (combination of all)
        final_state = hashlib.sha256()
        for k in sorted(field_checksums.keys()):
            final_state.update(field_checksums[k].encode('utf-8'))
        final_state.update(mixture_checksum.encode('utf-8'))
        final_state.update(registry_checksum.encode('utf-8'))
        
        result = {
            'engine_version': get_git_sha(),
            'world_law_version': cfg.world_law_version,
            'config_hash': hash_config(cfg),
            'seed': seed,
            'num_ticks': num_ticks,
            'initial_checksums': initial_checksums,
            'final_field_checksums': field_checksums,
            'mixture_checksum': mixture_checksum,
            'registry_checksum': registry_checksum,
            'final_state_checksum': final_state.hexdigest(),
            'summary': {
                'total_mass': float(fabric.rho.sum() + fabric.BH_mass.sum()),
                'total_energy': float(fabric.E_heat.sum() + fabric.E_rad.sum()),
                'total_microticks': quanta.total_microticks,
            }
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Generate golden checksums for Stratum")
    parser.add_argument('--seed', type=int, default=42, help='Base seed')
    parser.add_argument('--ticks', type=int, default=100, help='Number of ticks')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs to verify')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print(f"=== Golden Checksum Harness ===")
    print(f"Seed: {args.seed}")
    print(f"Ticks: {args.ticks}")
    print(f"Runs: {args.runs}")
    print()
    
    results = []
    for run in range(args.runs):
        print(f"--- Run {run + 1} ---")
        result = run_golden_checksum(args.seed, args.ticks, args.verbose)
        results.append(result)
        print(f"Final state checksum: {result['final_state_checksum']}")
        print(f"Summary: mass={result['summary']['total_mass']:.6f}, energy={result['summary']['total_energy']:.6f}")
        print()
    
    # Verify all runs produced identical checksums
    all_match = all(r['final_state_checksum'] == results[0]['final_state_checksum'] for r in results)
    
    print("=== Verification ===")
    if all_match:
        print("✅ PASS: All runs produced identical checksums")
        print(f"   Final state: {results[0]['final_state_checksum']}")
    else:
        print("❌ FAIL: Checksums differ between runs")
        for i, r in enumerate(results):
            print(f"   Run {i+1}: {r['final_state_checksum']}")
    
    # Output full result as JSON
    print()
    print("=== Full Result (Run 1) ===")
    print(json.dumps(results[0], indent=2))
    
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
