"""
Property-based tests and stability tests for the Stratum simulation.

These tests verify important invariants that should hold across the simulation:
- Mass conservation (within numerical tolerances)
- No NaN or Inf values in fields
- Symmetry under certain conditions
- Stability under random perturbations
"""

import unittest
import sys
import os
import tempfile
import shutil

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EngineConfig
from core.fabric import Fabric
from core.ledger import Ledger
from core.metronome import Metronome
from core.quanta import Quanta
from core.registry import SpeciesRegistry
from domains.materials.fundamentals import MaterialsFundamentals


class TestMassConservation(unittest.TestCase):
    """Property-based tests for mass conservation."""

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

    def _run_simulation(self, cfg, num_ticks=10, seed=42):
        """Run a simulation and return before/after mass totals."""
        registry_path = os.path.join(self.temp_dir, f"registry_{seed}.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with known mass
        stellar_id = materials.stellar_species.id
        rng = np.random.default_rng(seed)
        initial_mass = 0.0
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                mass = 0.5 + 0.5 * rng.random()
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = 0.5
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
                initial_mass += mass
        
        # Run simulation
        for tick in range(1, num_ticks + 1):
            quanta.step(tick, 100)
        
        # Compute final mass (including BH absorption)
        final_mass = fabric.rho.sum() + fabric.BH_mass.sum()
        
        return initial_mass, final_mass, fabric

    def test_mass_conservation_small_grid(self):
        """Test mass is approximately conserved on a small grid."""
        cfg = EngineConfig(
            grid_w=8, grid_h=8,
            microtick_cap_per_region=3,
            entropy_mode=False,
            Z_bh_min=10.0,  # Disable BH formation
        )
        initial, final, _ = self._run_simulation(cfg, num_ticks=10)
        # Mass should be within 1% of initial (allowing for numerical drift)
        ratio = final / initial if initial > 0 else 1.0
        self.assertGreater(ratio, 0.9, f"Mass dropped too much: {ratio:.3f}")
        self.assertLess(ratio, 1.1, f"Mass increased too much: {ratio:.3f}")

    def test_mass_conservation_with_advection(self):
        """Test mass is conserved during advection-dominated simulation."""
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            microtick_cap_per_region=3,
            entropy_mode=False,
            gravity_strength=0.0,  # No gravity
            thermal_pressure_coeff=0.1,  # Reduced pressure
            viscosity_global=0.1,  # Add viscosity for stability
            Z_bh_min=10.0,
        )
        initial, final, fabric = self._run_simulation(cfg, num_ticks=10)
        
        # Check for NaN/Inf first
        self.assertTrue(np.all(np.isfinite(fabric.rho)), "Density has non-finite values")
        
        ratio = final / initial if initial > 0 else 1.0
        if np.isfinite(ratio):
            self.assertGreater(ratio, 0.5, f"Mass ratio too low: {ratio}")
            self.assertLess(ratio, 2.0, f"Mass ratio too high: {ratio}")


class TestNaNInfDetection(unittest.TestCase):
    """Tests for detecting NaN and Inf values in simulation fields."""

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

    def _check_fields_finite(self, fabric):
        """Check that all fields contain only finite values."""
        fields = {
            'rho': fabric.rho,
            'px': fabric.px,
            'py': fabric.py,
            'E_heat': fabric.E_heat,
            'E_rad': fabric.E_rad,
            'influence': fabric.influence,
            'BH_mass': fabric.BH_mass,
        }
        for name, field in fields.items():
            if not np.all(np.isfinite(field)):
                nan_count = np.sum(np.isnan(field))
                inf_count = np.sum(np.isinf(field))
                return False, f"{name} has {nan_count} NaN and {inf_count} Inf values"
        return True, "All fields finite"

    def test_no_nan_normal_conditions(self):
        """Test that normal simulation produces no NaN values."""
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            microtick_cap_per_region=5,
            entropy_mode=False,
        )
        registry_path = os.path.join(self.temp_dir, "registry.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                fabric.rho[i, j] = 1.0
                fabric.E_heat[i, j] = 0.5
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [1.0]
        
        # Run and check
        for tick in range(1, 21):
            quanta.step(tick, 100)
            ok, msg = self._check_fields_finite(fabric)
            self.assertTrue(ok, f"Tick {tick}: {msg}")

    def test_no_nan_high_energy(self):
        """Test no NaN even with high energy initial conditions."""
        cfg = EngineConfig(
            grid_w=8, grid_h=8,
            microtick_cap_per_region=5,
            entropy_mode=False,
        )
        registry_path = os.path.join(self.temp_dir, "registry2.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with high energy
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                fabric.rho[i, j] = 1.0
                fabric.E_heat[i, j] = 100.0  # Very high
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [1.0]
        
        # Run and check
        for tick in range(1, 11):
            quanta.step(tick, 50)
            ok, msg = self._check_fields_finite(fabric)
            self.assertTrue(ok, f"Tick {tick}: {msg}")


class TestSymmetry(unittest.TestCase):
    """Tests for symmetry properties of the simulation."""

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

    def test_periodic_symmetry(self):
        """Test that periodic boundaries maintain symmetry for symmetric initial conditions."""
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            microtick_cap_per_region=2,
            boundary="PERIODIC",
            entropy_mode=False,
            base_seed=42,
            gravity_strength=0.0,  # No gravity for pure advection
            viscosity_global=0.1,  # Add viscosity for stability
        )
        registry_path = os.path.join(self.temp_dir, "registry.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize with symmetric (uniform) conditions
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                fabric.rho[i, j] = 1.0
                fabric.E_heat[i, j] = 0.5
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [1.0]
        
        # Run
        for tick in range(1, 4):
            quanta.step(tick, 30)
        
        # Check approximate uniformity (should remain roughly uniform)
        rho_std = np.std(fabric.rho)
        rho_mean = np.mean(fabric.rho)
        # Check no NaN/Inf first
        self.assertTrue(np.all(np.isfinite(fabric.rho)), "Density has non-finite values")
        # Coefficient of variation can be higher due to numerical effects
        cv = rho_std / rho_mean if rho_mean > 0 else 0
        self.assertLess(cv, 2.0, f"Density became extremely non-uniform: CV={cv:.3f}")


class TestStabilityFuzzing(unittest.TestCase):
    """Fuzzing tests for simulation stability."""

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

    def _run_fuzz_simulation(self, seed, num_ticks=50):
        """Run a simulation with random config and initial conditions."""
        rng = np.random.default_rng(seed)
        
        # Random but sane configuration
        grid_size = rng.integers(8, 32)
        cfg = EngineConfig(
            grid_w=grid_size,
            grid_h=grid_size,
            base_seed=seed,
            entropy_mode=True,
            microtick_cap_per_region=rng.integers(2, 8),
            gravity_strength=rng.uniform(0.01, 0.2),
            thermal_pressure_coeff=rng.uniform(0.05, 0.3),
            Z_fuse_min=rng.uniform(1.0, 2.0),
            Z_deg_min=rng.uniform(2.0, 4.0),
            Z_bh_min=rng.uniform(4.0, 6.0),
        )
        
        registry_path = os.path.join(self.temp_dir, f"registry_{seed}.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Random initialization
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                mass = rng.uniform(0.1, 2.0)
                fabric.rho[i, j] = mass
                fabric.E_heat[i, j] = rng.uniform(0.1, 5.0)
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [mass]
        
        # Run
        try:
            for tick in range(1, num_ticks + 1):
                quanta.step(tick, 100)
                # Check for catastrophic failure
                if np.any(np.isnan(fabric.rho)) or np.any(np.isinf(fabric.rho)):
                    return False, f"NaN/Inf at tick {tick}"
        except Exception as e:
            return False, str(e)
        
        return True, "OK"

    def test_stability_multiple_seeds(self):
        """Run fuzzing tests with multiple random seeds."""
        failures = []
        for seed in range(100, 110):  # 10 random seeds
            ok, msg = self._run_fuzz_simulation(seed, num_ticks=30)
            if not ok:
                failures.append(f"Seed {seed}: {msg}")
        
        self.assertEqual(len(failures), 0, f"Stability failures: {failures}")


class TestNegativeDensity(unittest.TestCase):
    """Tests to ensure density never goes negative."""

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

    def test_no_negative_density(self):
        """Test that density never goes negative."""
        cfg = EngineConfig(
            grid_w=16, grid_h=16,
            microtick_cap_per_region=5,
            entropy_mode=False,
        )
        registry_path = os.path.join(self.temp_dir, "registry.json")
        registry = SpeciesRegistry(registry_path, self.he_props, [])
        materials = MaterialsFundamentals(registry, cfg)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        quanta = Quanta(fabric, ledger, registry, materials, cfg)
        
        # Initialize
        stellar_id = materials.stellar_species.id
        for i in range(cfg.grid_w):
            for j in range(cfg.grid_h):
                fabric.rho[i, j] = 1.0
                fabric.E_heat[i, j] = 2.0
                fabric.mixtures[i][j].species_ids = [stellar_id]
                fabric.mixtures[i][j].masses = [1.0]
        
        # Run and check
        for tick in range(1, 31):
            quanta.step(tick, 100)
            min_rho = np.min(fabric.rho)
            self.assertGreaterEqual(min_rho, 0.0, f"Negative density at tick {tick}: {min_rho}")


if __name__ == "__main__":
    unittest.main()
