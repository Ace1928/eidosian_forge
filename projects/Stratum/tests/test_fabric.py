"""
Tests for the core.fabric module.

This module tests the Fabric class including field storage, mixture handling,
and spatial operations like gradients and divergence.
"""

import unittest
import sys
import os

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fabric import Fabric, Mixture
from core.config import EngineConfig


class TestMixture(unittest.TestCase):
    """Tests for the Mixture class."""

    def test_empty_mixture(self):
        """Test empty mixture initialization."""
        mix = Mixture([], [])
        self.assertEqual(mix.total_mass(), 0.0)
        self.assertEqual(len(mix.species_ids), 0)
        self.assertEqual(len(mix.masses), 0)

    def test_total_mass(self):
        """Test total_mass calculation."""
        mix = Mixture(["A", "B", "C"], [1.0, 2.0, 3.0])
        self.assertEqual(mix.total_mass(), 6.0)

    def test_normalise(self):
        """Test mixture normalization."""
        mix = Mixture(["A", "B"], [2.0, 4.0])
        mix.normalise(3.0)
        self.assertAlmostEqual(mix.total_mass(), 3.0, places=10)
        self.assertAlmostEqual(mix.masses[0], 1.0, places=10)
        self.assertAlmostEqual(mix.masses[1], 2.0, places=10)

    def test_normalise_zero_total(self):
        """Test normalization with zero total does nothing."""
        mix = Mixture(["A"], [0.0])
        mix.normalise(1.0)
        self.assertEqual(mix.masses[0], 0.0)

    def test_normalise_zero_target(self):
        """Test normalization to zero target does nothing."""
        mix = Mixture(["A"], [1.0])
        mix.normalise(0.0)
        self.assertEqual(mix.masses[0], 1.0)

    def test_get_weighted_property(self):
        """Test weighted property calculation."""
        mix = Mixture(["A", "B"], [1.0, 3.0])
        prop_table = {"A": 10.0, "B": 20.0}
        result = mix.get_weighted_property(prop_table, "prop")
        # (1*10 + 3*20) / 4 = 70/4 = 17.5
        self.assertAlmostEqual(result, 17.5, places=10)

    def test_get_weighted_property_missing_species(self):
        """Test weighted property with missing species returns 0."""
        mix = Mixture(["A", "C"], [1.0, 1.0])
        prop_table = {"A": 10.0}
        result = mix.get_weighted_property(prop_table, "prop")
        # (1*10 + 1*0) / 2 = 5.0
        self.assertAlmostEqual(result, 5.0, places=10)

    def test_add_species_mass_existing(self):
        """Test adding mass to existing species."""
        mix = Mixture(["A"], [1.0])
        mix.add_species_mass("A", 2.0, max_k=4)
        self.assertEqual(mix.masses[0], 3.0)

    def test_add_species_mass_new(self):
        """Test adding a new species when space available."""
        mix = Mixture(["A"], [1.0])
        mix.add_species_mass("B", 2.0, max_k=4)
        self.assertEqual(len(mix.species_ids), 2)
        self.assertIn("B", mix.species_ids)

    def test_add_species_mass_replace_smallest(self):
        """Test adding species replaces smallest when at max_k."""
        mix = Mixture(["A", "B"], [1.0, 2.0])
        mix.add_species_mass("C", 3.0, max_k=2)
        self.assertEqual(len(mix.species_ids), 2)
        # C should replace A (the smallest)
        self.assertIn("C", mix.species_ids)
        self.assertIn("B", mix.species_ids)
        self.assertNotIn("A", mix.species_ids)

    def test_add_species_mass_dont_replace_larger(self):
        """Test adding species doesn't replace if smaller than all."""
        mix = Mixture(["A", "B"], [5.0, 10.0])
        mix.add_species_mass("C", 1.0, max_k=2)
        # C is smaller than all, should not be added
        self.assertNotIn("C", mix.species_ids)

    def test_add_species_mass_zero(self):
        """Test adding zero mass does nothing."""
        mix = Mixture(["A"], [1.0])
        mix.add_species_mass("B", 0.0, max_k=4)
        self.assertEqual(len(mix.species_ids), 1)

    def test_cleanup_removes_negligible(self):
        """Test cleanup removes negligible masses."""
        mix = Mixture(["A", "B", "C"], [1.0, 1e-7, 2.0])
        mix.cleanup(eps=1e-6, max_k=4)
        self.assertEqual(len(mix.species_ids), 2)
        self.assertIn("A", mix.species_ids)
        self.assertIn("C", mix.species_ids)
        self.assertNotIn("B", mix.species_ids)

    def test_cleanup_trims_to_max_k(self):
        """Test cleanup keeps only top max_k species."""
        mix = Mixture(["A", "B", "C", "D"], [4.0, 3.0, 2.0, 1.0])
        mix.cleanup(eps=1e-6, max_k=2)
        self.assertEqual(len(mix.species_ids), 2)
        # Should keep the largest two
        self.assertIn("A", mix.species_ids)
        self.assertIn("B", mix.species_ids)

    def test_array_mode_add_and_cleanup(self):
        """Test array-backed mixture behavior."""
        mix = Mixture([], [], max_k=3)
        mix.add_species_mass("A", 1.0, max_k=3)
        mix.add_species_mass("B", 2.0, max_k=3)
        mix.add_species_mass("C", 0.5, max_k=3)
        self.assertEqual(mix.total_mass(), 3.5)
        mix.cleanup(eps=0.6, max_k=3)
        self.assertEqual(len(mix.species_ids), 2)
        self.assertIn("A", mix.species_ids)
        self.assertIn("B", mix.species_ids)


class TestFabric(unittest.TestCase):
    """Tests for the Fabric class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        self.fabric = Fabric(self.cfg)

    def test_field_shapes(self):
        """Test that all fields have correct shapes."""
        self.assertEqual(self.fabric.rho.shape, (8, 8))
        self.assertEqual(self.fabric.px.shape, (8, 8))
        self.assertEqual(self.fabric.py.shape, (8, 8))
        self.assertEqual(self.fabric.E_heat.shape, (8, 8))
        self.assertEqual(self.fabric.E_rad.shape, (8, 8))
        self.assertEqual(self.fabric.influence.shape, (8, 8))
        self.assertEqual(self.fabric.BH_mass.shape, (8, 8))
        self.assertEqual(self.fabric.EH_mask.shape, (8, 8))

    def test_field_initial_values(self):
        """Test that fields are initialized to zero."""
        self.assertTrue(np.all(self.fabric.rho == 0.0))
        self.assertTrue(np.all(self.fabric.px == 0.0))
        self.assertTrue(np.all(self.fabric.py == 0.0))
        self.assertTrue(np.all(self.fabric.E_heat == 0.0))
        self.assertTrue(np.all(self.fabric.E_rad == 0.0))

    def test_mixtures_structure(self):
        """Test that mixtures are initialized correctly."""
        self.assertEqual(len(self.fabric.mixtures), 8)
        self.assertEqual(len(self.fabric.mixtures[0]), 8)
        # Each mixture should be empty
        for i in range(8):
            for j in range(8):
                self.assertEqual(len(self.fabric.mixtures[i][j].species_ids), 0)

    def test_reset_influence(self):
        """Test reset_influence clears the influence field."""
        self.fabric.influence.fill(1.0)
        self.fabric.reset_influence()
        self.assertTrue(np.all(self.fabric.influence == 0.0))

    def test_reset_event_horizon(self):
        """Test reset_event_horizon clears the EH_mask field."""
        self.fabric.EH_mask.fill(1.0)
        self.fabric.reset_event_horizon()
        self.assertTrue(np.all(self.fabric.EH_mask == 0.0))

    def test_boundary_coord_periodic(self):
        """Test boundary_coord with PERIODIC boundary."""
        self.cfg.boundary = "PERIODIC"
        # Normal case
        self.assertEqual(self.fabric.boundary_coord(3, 4), (3, 4))
        # Wrap around positive
        self.assertEqual(self.fabric.boundary_coord(10, 5), (2, 5))
        self.assertEqual(self.fabric.boundary_coord(5, 12), (5, 4))
        # Wrap around negative
        self.assertEqual(self.fabric.boundary_coord(-1, 5), (7, 5))
        self.assertEqual(self.fabric.boundary_coord(5, -1), (5, 7))

    def test_boundary_coord_reflective(self):
        """Test boundary_coord with REFLECTIVE boundary."""
        self.cfg.boundary = "REFLECTIVE"
        # Normal case
        self.assertEqual(self.fabric.boundary_coord(3, 4), (3, 4))
        # Clamp at boundaries
        self.assertEqual(self.fabric.boundary_coord(10, 5), (7, 5))
        self.assertEqual(self.fabric.boundary_coord(5, 12), (5, 7))
        self.assertEqual(self.fabric.boundary_coord(-1, 5), (0, 5))
        self.assertEqual(self.fabric.boundary_coord(5, -1), (5, 0))

    def test_boundary_coord_open(self):
        """Test boundary_coord with OPEN boundary."""
        self.cfg.boundary = "OPEN"
        # Returns indices as-is
        self.assertEqual(self.fabric.boundary_coord(10, 5), (10, 5))
        self.assertEqual(self.fabric.boundary_coord(-1, 5), (-1, 5))

    def test_gradient_scalar_uniform(self):
        """Test gradient of uniform field is zero."""
        field = np.ones((8, 8))
        grad_x, grad_y = self.fabric.gradient_scalar(field)
        self.assertTrue(np.allclose(grad_x, 0.0))
        self.assertTrue(np.allclose(grad_y, 0.0))

    def test_gradient_scalar_linear_x(self):
        """Test gradient of linear field in x direction."""
        field = np.zeros((8, 8))
        for i in range(8):
            field[i, :] = i
        grad_x, grad_y = self.fabric.gradient_scalar(field)
        # Interior should have gradient of 1 in x direction
        self.assertTrue(np.allclose(grad_x[1:-1, :], 1.0))
        self.assertTrue(np.allclose(grad_y[:, 1:-1], 0.0))

    def test_gradient_scalar_linear_y(self):
        """Test gradient of linear field in y direction."""
        field = np.zeros((8, 8))
        for j in range(8):
            field[:, j] = j
        grad_x, grad_y = self.fabric.gradient_scalar(field)
        # Interior should have gradient of 1 in y direction
        self.assertTrue(np.allclose(grad_x[1:-1, :], 0.0))
        self.assertTrue(np.allclose(grad_y[:, 1:-1], 1.0))

    def test_divergence_vector_uniform(self):
        """Test divergence of uniform vector field is zero."""
        vx = np.ones((8, 8))
        vy = np.ones((8, 8))
        div = self.fabric.divergence_vector(vx, vy)
        self.assertTrue(np.allclose(div, 0.0))

    def test_divergence_vector_radial(self):
        """Test divergence of simple expanding field."""
        vx = np.zeros((8, 8))
        vy = np.zeros((8, 8))
        # Create a field that increases in x direction
        for i in range(8):
            vx[i, :] = i
        div = self.fabric.divergence_vector(vx, vy)
        # Interior cells should have positive divergence
        self.assertTrue(np.all(div[1:-1, 1:-1] > 0))


class TestFabricLargerGrid(unittest.TestCase):
    """Tests for Fabric with larger grid sizes."""

    def test_large_grid_creation(self):
        """Test that larger grids can be created."""
        cfg = EngineConfig(grid_w=64, grid_h=64)
        fabric = Fabric(cfg)
        self.assertEqual(fabric.rho.shape, (64, 64))

    def test_rectangular_grid(self):
        """Test non-square grid dimensions."""
        cfg = EngineConfig(grid_w=32, grid_h=16)
        fabric = Fabric(cfg)
        self.assertEqual(fabric.rho.shape, (32, 16))
        self.assertEqual(len(fabric.mixtures), 32)
        self.assertEqual(len(fabric.mixtures[0]), 16)


if __name__ == "__main__":
    unittest.main()
