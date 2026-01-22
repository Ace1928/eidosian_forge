#!/usr/bin/env python3
"""
Cross-process determinism test.

This test spawns separate Python processes to verify that the simulation
produces identical results across fresh interpreter invocations. This is
critical because Python's hash() function varies between processes unless
PYTHONHASHSEED is set.

The test runs:
1. Two simulations in separate processes with the same seed
2. Compares the resulting state checksums
3. Verifies they are bitwise identical
"""

import subprocess
import sys
import os
import json
import tempfile
import unittest


# Script to run in subprocess
SUBPROCESS_SCRIPT = '''
import sys
import os
import tempfile
import hashlib
import json

# Add the repository root to path
REPO_ROOT = "{repo_root}"
sys.path.insert(0, REPO_ROOT)

import numpy as np

from core.config import EngineConfig, DeterminismMode
from core.fabric import Fabric
from core.ledger import Ledger
from core.quanta import Quanta
from core.registry import SpeciesRegistry
from domains.materials.fundamentals import MaterialsFundamentals

def hash_array(arr):
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()

def run_simulation(seed, num_ticks, registry_path):
    he_props = [
        "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
        "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
    ]
    
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
    
    registry = SpeciesRegistry(registry_path, he_props, [])
    materials = MaterialsFundamentals(registry, cfg)
    fabric = Fabric(cfg)
    ledger = Ledger(fabric, cfg)
    quanta = Quanta(fabric, ledger, registry, materials, cfg)
    
    # Initialize deterministically
    stellar_id = materials.stellar_species.id
    for i in range(cfg.grid_w):
        for j in range(cfg.grid_h):
            mass = 0.5 + 0.5 * ((i + j + seed) % 100) / 100.0
            fabric.rho[i, j] = mass
            fabric.E_heat[i, j] = 0.5 + 0.25 * ((i * j + seed) % 50) / 50.0
            fabric.mixtures[i][j].species_ids = [stellar_id]
            fabric.mixtures[i][j].masses = [mass]
    
    # Run simulation
    for tick in range(1, num_ticks + 1):
        quanta.step(tick, 100)
    
    # Compute final state hash
    h = hashlib.sha256()
    h.update(hash_array(fabric.rho).encode())
    h.update(hash_array(fabric.px).encode())
    h.update(hash_array(fabric.py).encode())
    h.update(hash_array(fabric.E_heat).encode())
    h.update(hash_array(fabric.E_rad).encode())
    h.update(hash_array(fabric.BH_mass).encode())
    
    return {{
        'checksum': h.hexdigest(),
        'total_mass': float(fabric.rho.sum() + fabric.BH_mass.sum()),
        'total_energy': float(fabric.E_heat.sum() + fabric.E_rad.sum()),
    }}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ticks', type=int, required=True)
    parser.add_argument('--registry', type=str, required=True)
    args = parser.parse_args()
    
    result = run_simulation(args.seed, args.ticks, args.registry)
    print(json.dumps(result))
'''


class TestCrossProcessDeterminism(unittest.TestCase):
    """Tests for determinism across fresh Python processes."""

    def test_fresh_process_identical_checksums(self):
        """Test that separate Python processes produce identical results."""
        # Create temp directory for this test
        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "run_sim.py")
        
        # Get the repository root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Write the subprocess script with repo root injected
        with open(script_path, 'w') as f:
            f.write(SUBPROCESS_SCRIPT.format(repo_root=repo_root))
        
        try:
            results = []
            
            # Run simulation in 2 separate processes
            for run_id in range(2):
                registry_path = os.path.join(temp_dir, f"registry_{run_id}.json")
                
                # Run in subprocess with PYTHONHASHSEED=random to prove we don't leak Python hash
                env = os.environ.copy()
                env['PYTHONHASHSEED'] = 'random'
                
                result = subprocess.run(
                    [
                        sys.executable, script_path,
                        '--seed', '42',
                        '--ticks', '50',
                        '--registry', registry_path
                    ],
                    capture_output=True,
                    text=True,
                    cwd=repo_root,
                    env=env
                )
                
                if result.returncode != 0:
                    self.fail(f"Subprocess failed: {result.stderr}")
                
                # Parse result
                output = result.stdout.strip()
                run_result = json.loads(output)
                results.append(run_result)
            
            # Verify checksums are identical
            self.assertEqual(
                results[0]['checksum'], results[1]['checksum'],
                f"Checksums differ across processes:\n"
                f"  Run 1: {results[0]['checksum']}\n"
                f"  Run 2: {results[1]['checksum']}"
            )
            
            # Verify masses are identical
            self.assertEqual(results[0]['total_mass'], results[1]['total_mass'])
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "run_sim.py")
        
        # Get the repository root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        with open(script_path, 'w') as f:
            f.write(SUBPROCESS_SCRIPT.format(repo_root=repo_root))
        
        try:
            results = []
            
            for seed in [42, 43]:
                registry_path = os.path.join(temp_dir, f"registry_seed{seed}.json")
                
                result = subprocess.run(
                    [
                        sys.executable, script_path,
                        '--seed', str(seed),
                        '--ticks', '30',
                        '--registry', registry_path
                    ],
                    capture_output=True,
                    text=True,
                    cwd=repo_root
                )
                
                if result.returncode != 0:
                    self.fail(f"Subprocess failed: {result.stderr}")
                
                output = result.stdout.strip()
                run_result = json.loads(output)
                results.append(run_result)
            
            # Different seeds should produce different results
            self.assertNotEqual(
                results[0]['checksum'], results[1]['checksum'],
                "Different seeds should produce different checksums"
            )
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
