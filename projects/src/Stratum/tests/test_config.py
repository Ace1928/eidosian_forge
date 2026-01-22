"""
Tests for the core.config module.

This module tests the EngineConfig dataclass and its methods.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EngineConfig


class TestEngineConfig(unittest.TestCase):
    """Tests for the EngineConfig class."""

    def test_default_values(self):
        """Test that EngineConfig has sensible default values."""
        cfg = EngineConfig()
        # Grid dimensions
        self.assertEqual(cfg.grid_w, 128)
        self.assertEqual(cfg.grid_h, 128)
        # Seeds
        self.assertEqual(cfg.base_seed, 42)
        self.assertFalse(cfg.entropy_mode)
        self.assertFalse(cfg.replay_mode)
        # Boundary
        self.assertEqual(cfg.boundary, "PERIODIC")
        # Speeds
        self.assertEqual(cfg.v_max, 5.0)
        self.assertEqual(cfg.v_influence, 2.0)
        self.assertEqual(cfg.v_radiation, 5.0)

    def test_custom_values(self):
        """Test EngineConfig with custom values."""
        cfg = EngineConfig(
            grid_w=64,
            grid_h=32,
            base_seed=100,
            entropy_mode=True,
            boundary="REFLECTIVE",
        )
        self.assertEqual(cfg.grid_w, 64)
        self.assertEqual(cfg.grid_h, 32)
        self.assertEqual(cfg.base_seed, 100)
        self.assertTrue(cfg.entropy_mode)
        self.assertEqual(cfg.boundary, "REFLECTIVE")

    def test_to_dict(self):
        """Test the to_dict method returns all fields."""
        cfg = EngineConfig(grid_w=32, grid_h=32)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["grid_w"], 32)
        self.assertEqual(d["grid_h"], 32)
        self.assertIn("base_seed", d)
        self.assertIn("boundary", d)
        self.assertIn("v_max", d)

    def test_extras_default_factory(self):
        """Test that extras defaults to empty dict."""
        cfg1 = EngineConfig()
        cfg2 = EngineConfig()
        # Should be independent instances
        cfg1.extras["key"] = "value"
        self.assertNotIn("key", cfg2.extras)

    def test_physics_coefficients(self):
        """Test physics coefficient defaults."""
        cfg = EngineConfig()
        self.assertEqual(cfg.gravity_strength, 0.05)
        self.assertEqual(cfg.eos_gamma, 2.0)
        self.assertEqual(cfg.thermal_pressure_coeff, 0.1)
        self.assertEqual(cfg.repulsion_k, 50.0)
        self.assertEqual(cfg.repulsion_n, 2.0)
        self.assertEqual(cfg.shock_k, 0.2)

    def test_z_thresholds(self):
        """Test Z regime threshold defaults."""
        cfg = EngineConfig()
        self.assertEqual(cfg.Z_fuse_min, 1.5)
        self.assertEqual(cfg.Z_deg_min, 3.0)
        self.assertEqual(cfg.Z_bh_min, 4.5)
        self.assertEqual(cfg.Z_abs_max, 6.0)
        self.assertEqual(cfg.Z_star_flip, 2.5)

    def test_mixture_settings(self):
        """Test mixture handling settings."""
        cfg = EngineConfig()
        self.assertEqual(cfg.mixture_top_k, 4)
        self.assertEqual(cfg.mixture_eps_merge, 1e-6)

    def test_chemistry_settings(self):
        """Test chemistry gating settings."""
        cfg = EngineConfig()
        self.assertEqual(cfg.chemistry_tick_ratio, 5)
        self.assertEqual(cfg.Z_chem_max, 1.0)
        self.assertEqual(cfg.T_chem_max, 0.5)

    def test_black_hole_settings(self):
        """Test black hole settings."""
        cfg = EngineConfig()
        self.assertEqual(cfg.EH_k, 0.5)
        self.assertEqual(cfg.BH_absorb_energy_scale, 0.1)

    def test_stability_coefficients(self):
        """Test stability function coefficients."""
        cfg = EngineConfig()
        self.assertEqual(cfg.stability_low_coeff, 1.0)
        self.assertEqual(cfg.stability_high_coeff, 1.0)
        self.assertEqual(cfg.stability_temp_coeff, 0.5)


if __name__ == "__main__":
    unittest.main()
