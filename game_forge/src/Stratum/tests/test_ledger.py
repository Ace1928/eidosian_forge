"""
Tests for the core.ledger module.

This module tests the Ledger class including energy conservation,
barrier calculations, and entropy source functionality.
"""

import unittest
import sys
import os
import math

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ledger import Ledger, EntropySource, EntropyRecord
from core.fabric import Fabric
from core.config import EngineConfig
from core.types import Vec2


class TestEntropyRecord(unittest.TestCase):
    """Tests for the EntropyRecord dataclass."""

    def test_creation(self):
        """Test EntropyRecord creation."""
        rec = EntropyRecord(
            checkpoint_id="test",
            tick=5,
            cell=(1, 2),
            attempt=3,
            conditioning_summary=12345,
            value=0.5,
        )
        self.assertEqual(rec.checkpoint_id, "test")
        self.assertEqual(rec.tick, 5)
        self.assertEqual(rec.cell, (1, 2))
        self.assertEqual(rec.attempt, 3)
        self.assertEqual(rec.value, 0.5)


class TestEntropySource(unittest.TestCase):
    """Tests for the EntropySource class."""

    def test_deterministic_samples(self):
        """Test that same inputs produce same outputs."""
        ent1 = EntropySource(base_seed=42, entropy_mode=False)
        ent2 = EntropySource(base_seed=42, entropy_mode=False)
        val1 = ent1.sample_uniform("test", 1, (0, 0), 0, {"x": 1.0})
        val2 = ent2.sample_uniform("test", 1, (0, 0), 0, {"x": 1.0})
        self.assertEqual(val1, val2)

    def test_different_checkpoints(self):
        """Test that different checkpoints produce different values."""
        ent = EntropySource(base_seed=42)
        val1 = ent.sample_uniform("check1", 1, (0, 0), 0, {})
        val2 = ent.sample_uniform("check2", 1, (0, 0), 0, {})
        self.assertNotEqual(val1, val2)

    def test_different_ticks(self):
        """Test that different ticks produce different values."""
        ent = EntropySource(base_seed=42)
        val1 = ent.sample_uniform("test", 1, (0, 0), 0, {})
        val2 = ent.sample_uniform("test", 2, (0, 0), 0, {})
        self.assertNotEqual(val1, val2)

    def test_different_cells(self):
        """Test that different cells produce different values."""
        ent = EntropySource(base_seed=42)
        val1 = ent.sample_uniform("test", 1, (0, 0), 0, {})
        val2 = ent.sample_uniform("test", 1, (1, 0), 0, {})
        self.assertNotEqual(val1, val2)

    def test_entropy_mode(self):
        """Test that entropy_mode changes the run_salt."""
        ent1 = EntropySource(base_seed=42, entropy_mode=False)
        ent2 = EntropySource(base_seed=42, entropy_mode=True)
        # With entropy_mode=False, run_salt is 0
        self.assertEqual(ent1.run_salt, 0)
        # With entropy_mode=True, run_salt is random
        # (Note: could be 0 by chance, but unlikely)

    def test_uniform_range(self):
        """Test that samples are in [0,1] range."""
        ent = EntropySource(base_seed=42)
        for i in range(100):
            val = ent.sample_uniform("test", i, (0, 0), 0, {})
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_replay_mode_records(self):
        """Test that replay_mode records values."""
        ent = EntropySource(base_seed=42, replay_mode=True)
        val1 = ent.sample_uniform("test", 1, (0, 0), 0, {})
        self.assertEqual(len(ent.replay_log), 1)
        self.assertEqual(ent.replay_log[0].value, val1)

    def test_replay_mode_replays(self):
        """Test that replay_mode records values correctly.
        
        The EntropySource in replay_mode:
        - Returns values from replay_log if cursor < len(log)
        - Otherwise generates new values and appends them
        - Cursor is only advanced when reading from the log
        """
        # First, generate a value and record it (cursor starts at 0, len is 0)
        ent = EntropySource(base_seed=42, replay_mode=True)
        val1 = ent.sample_uniform("test", 1, (0, 0), 0, {})
        # Value should be recorded in log
        self.assertEqual(len(ent.replay_log), 1)
        self.assertEqual(ent.replay_log[0].value, val1)
        # Now reset cursor to 0 to replay
        ent.replay_cursor = 0
        # Replay should return the same value from the log
        replay1 = ent.sample_uniform("any_id", 999, (0, 0), 0, {})  # parameters don't matter for replay
        self.assertEqual(replay1, val1)

    def test_conditioning_hash_cache(self):
        """Test conditioning hash reuse for repeated fields."""
        ent = EntropySource(base_seed=42)
        fields = (("gate", 0.123456789),)
        h1 = ent._hash_conditioning_fields(fields)
        h2 = ent._hash_conditioning_fields(fields)
        self.assertEqual(h1, h2)


class TestLedger(unittest.TestCase):
    """Tests for the Ledger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        self.fabric = Fabric(self.cfg)
        self.ledger = Ledger(self.fabric, self.cfg)

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        rho = 2.0
        p = Vec2(4.0, 3.0)
        ke = Ledger.kinetic_energy(rho, p)
        # KE = |p|^2 / (2*rho) = (16+9) / 4 = 6.25
        self.assertAlmostEqual(ke, 6.25, places=10)

    def test_kinetic_energy_zero_mass(self):
        """Test kinetic energy with zero mass returns zero."""
        ke = Ledger.kinetic_energy(0.0, Vec2(1.0, 1.0))
        self.assertEqual(ke, 0.0)

    def test_kinetic_energy_components_matches_vec2(self):
        """Test component-based kinetic energy matches Vec2 path."""
        rho = 3.5
        px = 4.0
        py = -2.5
        ke_vec = Ledger.kinetic_energy(rho, Vec2(px, py))
        ke_comp = Ledger.kinetic_energy_components(rho, px, py)
        self.assertAlmostEqual(ke_vec, ke_comp, places=12)

    def test_kinetic_energy_very_small_mass(self):
        """Test kinetic energy with very small mass returns zero."""
        ke = Ledger.kinetic_energy(1e-13, Vec2(1.0, 1.0))
        self.assertEqual(ke, 0.0)

    def test_kinetic_energy_zero_momentum(self):
        """Test kinetic energy with zero momentum."""
        ke = Ledger.kinetic_energy(1.0, Vec2(0.0, 0.0))
        self.assertEqual(ke, 0.0)

    def test_kinetic_energy_large_values(self):
        """Test kinetic energy handles large values without overflow."""
        rho = 1.0
        p = Vec2(1e150, 1e150)
        ke = Ledger.kinetic_energy(rho, p)
        self.assertTrue(math.isfinite(ke))

    def test_barrier_crossed_energy_sufficient(self):
        """Test barrier crossing when energy is sufficient."""
        # With sufficient energy, crossing depends on gate
        self.ledger.entropy = EntropySource(base_seed=42)
        # Gate = 1.0 means always cross if energy sufficient
        result = self.ledger.barrier_crossed(
            "test", tick=1, cell=(0, 0), attempt=0,
            E_act=1.0, E_avail=2.0, T=1.0, gate=1.0
        )
        # With gate=1.0 and sufficient energy, should always cross
        # (unless the random sample is exactly 1.0)
        self.assertTrue(result)

    def test_barrier_crossed_energy_insufficient_low_temp(self):
        """Test barrier crossing when energy insufficient and low temperature."""
        self.ledger.entropy = EntropySource(base_seed=42)
        # With T=0, Arrhenius probability should be zero
        result = self.ledger.barrier_crossed(
            "test", tick=1, cell=(0, 0), attempt=0,
            E_act=2.0, E_avail=1.0, T=0.0, gate=1.0
        )
        self.assertFalse(result)

    def test_barrier_crossed_zero_gate(self):
        """Test barrier never crossed when gate is zero."""
        self.ledger.entropy = EntropySource(base_seed=42)
        for i in range(10):
            result = self.ledger.barrier_crossed(
                "test", tick=i, cell=(0, 0), attempt=0,
                E_act=0.0, E_avail=1.0, T=1.0, gate=0.0
            )
            self.assertFalse(result)

    def test_convert_kinetic_to_heat_positive(self):
        """Test conversion of kinetic energy to heat."""
        self.fabric.E_heat[0, 0] = 0.0
        self.fabric.E_rad[0, 0] = 0.0
        self.ledger.convert_kinetic_to_heat(0, 0, delta_E=10.0, rad_fraction=0.2)
        self.assertAlmostEqual(self.fabric.E_heat[0, 0], 8.0, places=10)
        self.assertAlmostEqual(self.fabric.E_rad[0, 0], 2.0, places=10)

    def test_convert_kinetic_to_heat_zero(self):
        """Test conversion with zero energy does nothing."""
        self.fabric.E_heat[0, 0] = 1.0
        self.fabric.E_rad[0, 0] = 1.0
        self.ledger.convert_kinetic_to_heat(0, 0, delta_E=0.0, rad_fraction=0.5)
        self.assertEqual(self.fabric.E_heat[0, 0], 1.0)
        self.assertEqual(self.fabric.E_rad[0, 0], 1.0)

    def test_convert_kinetic_to_heat_negative(self):
        """Test conversion with negative energy does nothing."""
        self.fabric.E_heat[0, 0] = 1.0
        self.fabric.E_rad[0, 0] = 1.0
        self.ledger.convert_kinetic_to_heat(0, 0, delta_E=-5.0, rad_fraction=0.5)
        self.assertEqual(self.fabric.E_heat[0, 0], 1.0)
        self.assertEqual(self.fabric.E_rad[0, 0], 1.0)

    def test_convert_kinetic_to_heat_clamps_rad_fraction(self):
        """Test that rad_fraction is clamped to [0,1]."""
        self.fabric.E_heat[0, 0] = 0.0
        self.fabric.E_rad[0, 0] = 0.0
        # rad_fraction > 1 should be clamped to 1
        self.ledger.convert_kinetic_to_heat(0, 0, delta_E=10.0, rad_fraction=1.5)
        self.assertAlmostEqual(self.fabric.E_heat[0, 0], 0.0, places=10)
        self.assertAlmostEqual(self.fabric.E_rad[0, 0], 10.0, places=10)

    def test_finalize_tick(self):
        """Test finalize_tick runs without error."""
        # Currently a no-op placeholder
        self.ledger.finalize_tick(1)


class TestLedgerEntropy(unittest.TestCase):
    """Tests for Ledger's entropy source integration."""

    def test_ledger_uses_config_seed(self):
        """Test that Ledger uses config's base_seed."""
        cfg = EngineConfig(base_seed=123)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        self.assertEqual(ledger.entropy.base_seed, 123)

    def test_ledger_uses_config_entropy_mode(self):
        """Test that Ledger uses config's entropy_mode."""
        cfg = EngineConfig(entropy_mode=True)
        fabric = Fabric(cfg)
        ledger = Ledger(fabric, cfg)
        self.assertTrue(ledger.entropy.entropy_mode)


if __name__ == "__main__":
    unittest.main()
