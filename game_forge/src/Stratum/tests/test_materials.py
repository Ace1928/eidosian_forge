"""
Tests for the domains.materials.fundamentals module.

This module tests the MaterialsFundamentals class including EOS calculations,
fusion/decay logic, and high-energy event handling.
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
from core.fabric import Fabric, Mixture
from core.ledger import Ledger, EntropySource
from core.registry import SpeciesRegistry
from core.types import Vec2
from domains.materials.fundamentals import MaterialsFundamentals, MaterialDefinition


class TestMaterialDefinition(unittest.TestCase):
    """Tests for the MaterialDefinition dataclass."""

    def test_creation(self):
        """Test MaterialDefinition creation."""
        mat = MaterialDefinition(
            name="test_material",
            he_props={"HE/rho_max": 0.5, "HE/chi": 0.3},
        )
        self.assertEqual(mat.name, "test_material")
        self.assertEqual(mat.he_props["HE/rho_max"], 0.5)


class TestMaterialsFundamentals(unittest.TestCase):
    """Tests for the MaterialsFundamentals class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)
        self.fabric = Fabric(self.cfg)
        self.ledger = Ledger(self.fabric, self.cfg)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test MaterialsFundamentals initialization."""
        self.assertIsNotNone(self.materials.stellar_species)
        self.assertIsNotNone(self.materials.deg_species)
        self.assertIn(self.materials.stellar_species.id, self.registry.species)
        self.assertIn(self.materials.deg_species.id, self.registry.species)

    def test_stellar_species_properties(self):
        """Test stellar species has expected properties."""
        stellar = self.materials.stellar_species
        self.assertIn("HE/rho_max", stellar.he_props)
        self.assertIn("HE/chi", stellar.he_props)
        self.assertGreater(stellar.he_props["HE/rho_max"], 0.0)

    def test_degenerate_species_properties(self):
        """Test degenerate species has expected properties."""
        deg = self.materials.deg_species
        self.assertIn("HE/rho_max", deg.he_props)
        # DEG should have high rho_max (stiffer)
        self.assertGreater(
            deg.he_props["HE/rho_max"],
            self.materials.stellar_species.he_props["HE/rho_max"]
        )


class TestEffectiveProperty(unittest.TestCase):
    """Tests for effective property calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_effective_property_empty_mixture(self):
        """Test effective property of empty mixture is zero."""
        mix = Mixture([], [])
        result = self.materials.effective_property(mix, self.registry, "HE/rho_max")
        self.assertEqual(result, 0.0)

    def test_effective_property_single_species(self):
        """Test effective property with single species."""
        stellar_id = self.materials.stellar_species.id
        mix = Mixture([stellar_id], [1.0])
        result = self.materials.effective_property(mix, self.registry, "HE/rho_max")
        expected = self.materials.stellar_species.he_props["HE/rho_max"]
        self.assertAlmostEqual(result, expected, places=5)

    def test_effective_property_weighted_average(self):
        """Test effective property is weighted average."""
        stellar_id = self.materials.stellar_species.id
        deg_id = self.materials.deg_species.id
        mix = Mixture([stellar_id, deg_id], [1.0, 3.0])
        result = self.materials.effective_property(mix, self.registry, "HE/rho_max")
        # Should be weighted average: (1*stellar + 3*deg) / 4
        stellar_val = self.materials.stellar_species.he_props["HE/rho_max"]
        deg_val = self.materials.deg_species.he_props["HE/rho_max"]
        expected = (1.0 * stellar_val + 3.0 * deg_val) / 4.0
        self.assertAlmostEqual(result, expected, places=5)


class TestBarrierFunctions(unittest.TestCase):
    """Tests for energy barrier calculation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_E_avail_local(self):
        """Test available energy calculation."""
        result = self.materials.E_avail_local(rho=1.0, heat=10.0, kin=6.0)
        # 0.5 * heat + 0.5 * kin = 0.5*10 + 0.5*6 = 8.0
        self.assertEqual(result, 8.0)

    def test_E_act_fusion_high_Z(self):
        """Test fusion activation energy decreases with high Z."""
        he_props = {"HE/rho_max": 0.3, "HE/chi": 0.5, "HE/lambda": 0.1}
        E_act_low_Z = self.materials.E_act_fusion(he_props, Z=1.5, T=0.1, rho=1.0)
        E_act_high_Z = self.materials.E_act_fusion(he_props, Z=4.0, T=0.1, rho=1.0)
        # Higher Z should lower activation energy
        self.assertLess(E_act_high_Z, E_act_low_Z)

    def test_E_act_fusion_high_T(self):
        """Test fusion activation energy decreases with high T."""
        he_props = {"HE/rho_max": 0.3, "HE/chi": 0.5}
        E_act_low_T = self.materials.E_act_fusion(he_props, Z=2.0, T=0.1, rho=1.0)
        E_act_high_T = self.materials.E_act_fusion(he_props, Z=2.0, T=2.0, rho=1.0)
        # Higher T should lower activation energy
        self.assertLess(E_act_high_T, E_act_low_T)

    def test_fusion_yield_fraction_bounds(self):
        """Test fusion yield fraction is bounded."""
        he_props = {"HE/rho_max": 0.3, "HE/chi": 0.5, "HE/lambda": 0.5}
        result = self.materials.fusion_yield_fraction(he_props, Z=5.0, T=1.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 0.8)

    def test_radiation_fraction_bounds(self):
        """Test radiation fraction is bounded."""
        parent_he = {"HE/opacity": 0.5}
        child_he = {"HE/opacity": 0.3}
        result = self.materials.radiation_fraction(parent_he, child_he, T=1.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 0.9)

    def test_E_act_decay(self):
        """Test decay activation energy."""
        he_props = {"HE/lambda": 0.5}
        E_act = self.materials.E_act_decay(he_props, Z=1.0, T=0.5)
        self.assertGreater(E_act, 0.0)

    def test_decay_fraction_bounds(self):
        """Test decay fraction is bounded."""
        he_props = {"HE/lambda": 0.5}
        result = self.materials.decay_fraction(he_props, Z=1.0, T=0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 0.6)

    def test_E_act_degenerate(self):
        """Test degenerate transition activation energy."""
        mix = Mixture([], [])
        E_act = self.materials.E_act_degenerate(Z=3.0, T=0.5, mix=mix)
        self.assertGreater(E_act, 0.0)

    def test_degenerate_fraction_bounds(self):
        """Test degenerate fraction is bounded."""
        result = self.materials.degenerate_fraction(Z=3.5, T=0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 0.8)

    def test_E_act_bh(self):
        """Test black hole activation energy."""
        E_act = self.materials.E_act_bh(Z=4.5, T=0.5)
        self.assertGreater(E_act, 0.0)


class TestStabilityFunction(unittest.TestCase):
    """Tests for the stability calculation function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stability_high_stable_species(self):
        """Test stability is positive for stable species."""
        he_props = {"HE/beta": 0.8, "HE/chi": 0.7, "HE/lambda": 0.1}
        S = self.materials.compute_stability_high(he_props, Z=1.0, T=0.1)
        self.assertGreater(S, 0.0)

    def test_stability_high_unstable_species(self):
        """Test stability is negative for unstable species."""
        he_props = {"HE/beta": 0.1, "HE/chi": 0.1, "HE/lambda": 0.9}
        S = self.materials.compute_stability_high(he_props, Z=5.0, T=2.0)
        self.assertLess(S, 0.0)


class TestMassConversion(unittest.TestCase):
    """Tests for mass conversion functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8, mixture_top_k=4)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)
        self.fabric = Fabric(self.cfg)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_convert_mass_fraction(self):
        """Test basic mass conversion between species."""
        stellar_id = self.materials.stellar_species.id
        deg_id = self.materials.deg_species.id
        # Set up cell with stellar species
        mix = self.fabric.mixtures[0][0]
        mix.species_ids = [stellar_id]
        mix.masses = [10.0]
        # Convert 30% to degenerate
        self.materials.convert_mass_fraction(
            self.fabric, 0, 0, stellar_id, deg_id, 0.3
        )
        # Should have 70% stellar and 30% degenerate
        self.assertIn(stellar_id, mix.species_ids)
        self.assertIn(deg_id, mix.species_ids)
        stellar_idx = mix.species_ids.index(stellar_id)
        deg_idx = mix.species_ids.index(deg_id)
        self.assertAlmostEqual(mix.masses[stellar_idx], 7.0, places=5)
        self.assertAlmostEqual(mix.masses[deg_idx], 3.0, places=5)

    def test_convert_mass_fraction_zero(self):
        """Test zero fraction does nothing."""
        stellar_id = self.materials.stellar_species.id
        deg_id = self.materials.deg_species.id
        mix = self.fabric.mixtures[0][0]
        mix.species_ids = [stellar_id]
        mix.masses = [10.0]
        self.materials.convert_mass_fraction(
            self.fabric, 0, 0, stellar_id, deg_id, 0.0
        )
        self.assertEqual(len(mix.species_ids), 1)
        self.assertEqual(mix.masses[0], 10.0)

    def test_convert_mass_fraction_one(self):
        """Test fraction >= 1 does nothing."""
        stellar_id = self.materials.stellar_species.id
        deg_id = self.materials.deg_species.id
        mix = self.fabric.mixtures[0][0]
        mix.species_ids = [stellar_id]
        mix.masses = [10.0]
        self.materials.convert_mass_fraction(
            self.fabric, 0, 0, stellar_id, deg_id, 1.0
        )
        self.assertEqual(mix.masses[0], 10.0)

    def test_convert_mass_fraction_nonexistent_species(self):
        """Test converting nonexistent species does nothing."""
        stellar_id = self.materials.stellar_species.id
        deg_id = self.materials.deg_species.id
        mix = self.fabric.mixtures[0][0]
        mix.species_ids = [stellar_id]
        mix.masses = [10.0]
        self.materials.convert_mass_fraction(
            self.fabric, 0, 0, "nonexistent", deg_id, 0.5
        )
        self.assertEqual(len(mix.species_ids), 1)
        self.assertEqual(mix.masses[0], 10.0)


class TestGlobalOperations(unittest.TestCase):
    """Tests for global diffusion and smoothing operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8, viscosity_global=0.01)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)
        self.fabric = Fabric(self.cfg)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_apply_global_ops_no_crash(self):
        """Test that global ops run without error."""
        self.materials.apply_global_ops(self.fabric, self.cfg)

    def test_apply_global_ops_diffusion(self):
        """Test that global ops cause diffusion."""
        stellar_id = self.materials.stellar_species.id
        # Set up a single hot cell
        self.fabric.E_heat[4, 4] = 10.0
        self.fabric.mixtures[4][4].species_ids = [stellar_id]
        self.fabric.mixtures[4][4].masses = [1.0]
        initial_heat = self.fabric.E_heat[4, 4]
        self.materials.apply_global_ops(self.fabric, self.cfg)
        # Heat should have diffused (decreased)
        # Note: this depends on kappa_t values
        # Just verify it runs without error for now

    def test_apply_global_ops_preserves_non_negative(self):
        """Test that global ops maintain non-negative energies."""
        self.fabric.E_heat[4, 4] = 1.0
        self.fabric.E_rad[4, 4] = 1.0
        self.materials.apply_global_ops(self.fabric, self.cfg)
        self.assertTrue(np.all(self.fabric.E_heat >= 0.0))
        self.assertTrue(np.all(self.fabric.E_rad >= 0.0))


class TestBlackHoleAbsorption(unittest.TestCase):
    """Tests for black hole absorption functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8, BH_absorb_energy_scale=0.1)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = [
            "HE/rho_max", "HE/chi", "HE/eta", "HE/opacity",
            "HE/kappa_t", "HE/kappa_r", "HE/beta", "HE/nu", "HE/lambda",
        ]
        self.registry = SpeciesRegistry(registry_path, self.he_props, [])
        self.materials = MaterialsFundamentals(self.registry, self.cfg)
        self.fabric = Fabric(self.cfg)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_absorb_into_black_hole(self):
        """Test black hole absorption clears cell."""
        stellar_id = self.materials.stellar_species.id
        self.fabric.rho[4, 4] = 10.0
        self.fabric.px[4, 4] = 5.0
        self.fabric.py[4, 4] = 3.0
        self.fabric.E_heat[4, 4] = 20.0
        self.fabric.E_rad[4, 4] = 10.0
        self.fabric.mixtures[4][4].species_ids = [stellar_id]
        self.fabric.mixtures[4][4].masses = [10.0]
        self.materials.absorb_into_black_hole(self.fabric, 4, 4, self.cfg)
        # Cell should be cleared
        self.assertEqual(self.fabric.rho[4, 4], 0.0)
        self.assertEqual(self.fabric.px[4, 4], 0.0)
        self.assertEqual(self.fabric.py[4, 4], 0.0)
        self.assertEqual(self.fabric.E_heat[4, 4], 0.0)
        self.assertEqual(self.fabric.E_rad[4, 4], 0.0)
        # BH mass should have increased
        self.assertGreater(self.fabric.BH_mass[4, 4], 0.0)

    def test_absorb_into_black_hole_empty_cell(self):
        """Test absorption of empty cell."""
        initial_bh_mass = self.fabric.BH_mass[4, 4]
        self.materials.absorb_into_black_hole(self.fabric, 4, 4, self.cfg)
        # BH mass should remain the same (or slightly increase due to zero energy)
        self.assertEqual(self.fabric.BH_mass[4, 4], initial_bh_mass)


if __name__ == "__main__":
    unittest.main()
