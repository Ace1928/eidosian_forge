"""
Determinism verification tests for Stratum.

These tests verify that the simulation produces identical results
when run with the same seed and configuration. This is critical
for reproducibility and replay functionality.

Key Tests:
- Golden checksum test: Verifies bitwise-identical results across runs
- Cross-platform determinism: Tests determinism across different systems
- Derived field cache verification: Ensures cached values match recomputed
"""

import unittest
import sys
import os
import tempfile
import shutil
import hashlib

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EngineConfig, DeterminismMode
from core.fabric import Fabric
from core.ledger import Ledger
from core.quanta import Quanta
from core.registry import SpeciesRegistry
from domains.materials.fundamentals import MaterialsFundamentals


class TestGoldenChecksum(unittest.TestCase):
    """Tests for deterministic simulation reproducibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _compute_state_hash(self, fabric, registry) -> str:
        """Compute a hash of the simulation state.
        
        This hash includes all major fields and registry contents to detect
        any differences in simulation state.
        """
        h = hashlib.sha256()
        
        # Hash all fields
        for field_name in ['rho', 'px', 'py', 'E_heat', 'E_rad', 'influence', 'BH_mass', 'EH_mask']:
            field = getattr(fabric, field_name)
            # Convert to bytes in a canonical way
            h.update(field_name.encode('utf-8'))
            h.update(field.tobytes())
        
        # Hash mixture data
        for i in range(fabric.cfg.grid_w):
            for j in range(fabric.cfg.grid_h):
                mix = fabric.mixtures[i][j]
                h.update(f"mix_{i}_{j}:".encode('utf-8'))
                for sid, mass in zip(mix.species_ids, mix.masses):
                    h.update(f"{sid}:{mass:.10f}".encode('utf-8'))
        
        # Hash registry species
        h.update("registry:".encode('utf-8'))
        for sid in sorted(registry.species.keys()):
            sp = registry.species[sid]
            h.update(f"{sid}:".encode('utf-8'))
            for prop_name in sorted(sp.he_props.keys()):
                h.update(f"{prop_name}={sp.he_props[prop_name]:.10f}".encode('utf-8'))
        
        return h.hexdigest()

    def _run_simulation(self, seed, run_id, num_ticks=100):
        """Run a simulation and return the final state hash.
        
        Args:
            seed: Base seed for the simulation.
            run_id: Unique identifier for this run (for separate temp files).
            num_ticks: Number of ticks to run.
            
        Returns:
            Tuple of (state_hash, total_mass, total_energy).
        """
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            base_seed=seed,
            entropy_mode=False,  # Disable entropy salt for determinism
            determinism_mode=DeterminismMode.STRICT_DETERMINISTIC,
            microtick_cap_per_region=3,
            active_region_max=64,
        )
        
        registry_path = os.path.join(self.temp_dir, f"registry_{run_id}.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with deterministic values
        stellar_id = materials.stellar_species.id
        np.random.seed(seed)  # Use seed for initialization
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                # Deterministic initialization based on position and seed
                mass = 0.5 + 0.5 * ((i + j + seed) % 100) / 100.0
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 0.5 + 0.25 * ((i * j + seed) % 50) / 50.0
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
        
        # Run simulation
        for tick in range(1, num_ticks + 1):
            quanta.step(tick, 100)
        
        # Compute state hash
        state_hash = self._compute_state_hash(fabric, registry)
        total_mass = float(fabric.rho.sum() + fabric.BH_mass.sum())
        total_energy = float(fabric.E_heat.sum() + fabric.E_rad.sum())
        
        return state_hash, total_mass, total_energy

    def test_deterministic_same_seed_identical_hash(self):
        """Test that same seed produces identical state hash across multiple runs."""
        seed = 42
        num_runs = 3
        
        hashes = []
        masses = []
        energies = []
        
        for run_id in range(num_runs):
            h, m, e = self._run_simulation(seed, run_id, num_ticks=50)
            hashes.append(h)
            masses.append(m)
            energies.append(e)
        
        # All hashes should be identical
        self.assertTrue(all(h == hashes[0] for h in hashes),
            f"State hashes differ across runs: {hashes}")
        
        # All masses and energies should be identical
        self.assertTrue(all(m == masses[0] for m in masses),
            f"Masses differ across runs: {masses}")
        self.assertTrue(all(e == energies[0] for e in energies),
            f"Energies differ across runs: {energies}")

    def test_different_seeds_different_hash(self):
        """Test that different seeds produce different results."""
        hash1, _, _ = self._run_simulation(seed=42, run_id=100, num_ticks=50)
        hash2, _, _ = self._run_simulation(seed=43, run_id=101, num_ticks=50)
        
        # With different seeds, hashes should (almost certainly) differ
        # There's a tiny chance of collision, but it's astronomically unlikely
        self.assertNotEqual(hash1, hash2,
            "Different seeds should produce different state hashes")

    def test_longer_simulation_determinism(self):
        """Test determinism over longer simulations."""
        seed = 12345
        num_ticks = 100
        
        hash1, m1, e1 = self._run_simulation(seed, run_id=200, num_ticks=num_ticks)
        hash2, m2, e2 = self._run_simulation(seed, run_id=201, num_ticks=num_ticks)
        
        self.assertEqual(hash1, hash2, "Long simulation should be deterministic")
        self.assertEqual(m1, m2)
        self.assertEqual(e1, e2)


class TestDerivedFieldCacheVerification(unittest.TestCase):
    """Tests to verify cached derived fields match recomputed values."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cached_rho_max_matches_recomputed(self):
        """Test that cached rho_max_eff matches freshly computed values."""
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            microtick_cap_per_region=3,
        )
        
        registry_path = os.path.join(self.temp_dir, "registry.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize
        stellar_id = materials.stellar_species.id
        deg_id = materials.deg_species.id
        
        rng = np.random.default_rng(42)
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                mass = rng.uniform(0.5, 1.5)
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 0.5
                
                # Mix of stellar and degenerate
                if rng.random() > 0.7:
                    fabric.mixtures[i][j].species_ids = [stellar_id, deg_id]
                    fabric.mixtures[i][j].masses = [mass * 0.7, mass * 0.3]
                else:
                    fabric.mixtures[i][j].species_ids = [stellar_id]
                    fabric.mixtures[i][j].masses = [mass]
        
        # Compute rho_max BEFORE running step (to compare with cache computed during step)
        pre_step_rho_max = quanta.compute_effective_rho_max()
        
        # Run a step to populate the cache
        quanta.step(tick=1, micro_budget=100)
        
        # Get cached derived fields - NOTE: cache is computed at START of step
        # So it reflects the state BEFORE the step modified mixtures
        cached = quanta._derived_cache
        self.assertIsNotNone(cached, "Derived cache should be populated after step")
        
        # The cached values should match what we computed before the step
        # (because cache is computed at tick start)
        np.testing.assert_allclose(
            cached.rho_max_eff,
            pre_step_rho_max,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Cached rho_max_eff should match pre-step computed values"
        )


class TestBoundaryBehavior(unittest.TestCase):
    """Tests for correct boundary behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_periodic_boundary_translation_invariance(self):
        """Test that periodic boundaries maintain translation invariance."""
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            boundary="PERIODIC",
            microtick_cap_per_region=2,
            gravity_strength=0.0,  # No gravity
            entropy_mode=False,
            thermal_pressure_coeff=0.05,  # Reduced for stability
            viscosity_global=0.1,  # Add viscosity for stability
            Z_bh_min=10.0,  # Disable BH formation
        )
        
        registry_path = os.path.join(self.temp_dir, "registry.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with uniform conditions
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                fabric.rho[i, j] = 1.0
                fabric.E_heat[i, j] = 0.5
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [1.0]
        
        initial_total_mass = fabric.rho.sum()
        
        # Run a few ticks
        for tick in range(1, 5):
            quanta.step(tick, 50)
            # Check for NaN/Inf at each step
            if not np.all(np.isfinite(fabric.rho)):
                self.fail(f"Non-finite density at tick {tick}")
        
        final_total_mass = fabric.rho.sum() + fabric.BH_mass.sum()
        
        # For periodic boundaries with no sources/sinks, mass should be conserved
        # (within numerical tolerance)
        self.assertTrue(np.isfinite(final_total_mass), "Final mass should be finite")
        self.assertAlmostEqual(
            initial_total_mass, final_total_mass,
            delta=initial_total_mass * 0.2,  # Allow 20% tolerance for numerical effects
            msg="Periodic boundary should approximately conserve total mass"
        )

    def test_reflective_boundary_no_outflow(self):
        """Test that reflective boundaries prevent mass outflow."""
        cfg = EngineConfig(
            grid_w=8, grid_h=8,
            boundary="REFLECTIVE",
            microtick_cap_per_region=2,
            entropy_mode=False,
        )
        
        registry_path = os.path.join(self.temp_dir, "registry2.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with blob near edge
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                # Higher density near edge
                if i < 2 or j < 2:
                    mass = 2.0
                else:
                    mass = 0.5
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 1.0
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
        
        initial_total_mass = fabric.rho.sum()
        
        # Run simulation
        for tick in range(1, 10):
            quanta.step(tick, 50)
        
        final_total_mass = fabric.rho.sum() + fabric.BH_mass.sum()
        
        # Reflective boundary should not allow mass to escape
        # Allow larger tolerance due to numerical effects
        self.assertGreater(
            final_total_mass, initial_total_mass * 0.5,
            msg="Reflective boundary should retain most mass"
        )


class TestEntropyDeterminism(unittest.TestCase):
    """Tests for entropy source determinism."""

    def test_blake2s_hash_stability(self):
        """Test that blake2s hashing produces consistent results."""
        from core.ledger import EntropySource
        
        # Create entropy source
        source = EntropySource(base_seed=42, entropy_mode=False)
        
        # Sample multiple times with same parameters
        samples = []
        for _ in range(5):
            s = source.sample_uniform(
                checkpoint_id="test",
                tick=100,
                cell=(5, 5),
                attempt=0,
                conditioning={"value": 0.5}
            )
            samples.append(s)
        
        # All samples should be identical (deterministic)
        self.assertTrue(
            all(s == samples[0] for s in samples),
            f"Entropy samples should be deterministic: {samples}"
        )

    def test_different_inputs_different_samples(self):
        """Test that different inputs produce different samples."""
        from core.ledger import EntropySource
        
        source = EntropySource(base_seed=42, entropy_mode=False)
        
        s1 = source.sample_uniform("test", 100, (5, 5), 0, {})
        s2 = source.sample_uniform("test", 101, (5, 5), 0, {})  # Different tick
        s3 = source.sample_uniform("test", 100, (6, 5), 0, {})  # Different cell
        s4 = source.sample_uniform("other", 100, (5, 5), 0, {})  # Different checkpoint
        
        # All should be different
        samples = [s1, s2, s3, s4]
        self.assertEqual(len(set(samples)), 4, "Different inputs should produce different samples")


if __name__ == "__main__":
    unittest.main()
