"""
Tests for the core.metronome module.

This module tests the Metronome class for timing and budget allocation.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metronome import Metronome, MetronomeStats
from core.config import EngineConfig


class TestMetronomeStats(unittest.TestCase):
    """Tests for the MetronomeStats dataclass."""

    def test_default_values(self):
        """Test MetronomeStats default values."""
        stats = MetronomeStats(tick=0)
        self.assertEqual(stats.tick, 0)
        self.assertEqual(stats.microticks_used, 0)
        self.assertEqual(stats.active_cells, 0)
        self.assertEqual(stats.energy_total, 0.0)

    def test_custom_values(self):
        """Test MetronomeStats with custom values."""
        stats = MetronomeStats(
            tick=5,
            microticks_used=100,
            active_cells=50,
            energy_total=1000.0,
        )
        self.assertEqual(stats.tick, 5)
        self.assertEqual(stats.microticks_used, 100)
        self.assertEqual(stats.active_cells, 50)
        self.assertEqual(stats.energy_total, 1000.0)


class TestMetronome(unittest.TestCase):
    """Tests for the Metronome class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = EngineConfig(grid_w=16, grid_h=16, microtick_cap_per_region=10)
        self.metronome = Metronome(self.cfg)

    def test_initialization(self):
        """Test Metronome initialization."""
        self.assertIsNotNone(self.metronome.cfg)
        self.assertIsInstance(self.metronome.stats, MetronomeStats)
        self.assertEqual(self.metronome.stats.tick, 0)

    def test_allocate_returns_dict(self):
        """Test that allocate returns a dictionary."""
        budgets = self.metronome.allocate(tick=1)
        self.assertIsInstance(budgets, dict)

    def test_allocate_contains_quanta(self):
        """Test that allocate returns quanta_micro_ops budget."""
        budgets = self.metronome.allocate(tick=1)
        self.assertIn("quanta_micro_ops", budgets)

    def test_allocate_budget_calculation(self):
        """Test that budget calculation is correct."""
        budgets = self.metronome.allocate(tick=1)
        expected = self.cfg.grid_w * self.cfg.grid_h * self.cfg.microtick_cap_per_region
        # 16 * 16 * 10 = 2560
        self.assertEqual(budgets["quanta_micro_ops"], expected)

    def test_allocate_updates_stats_tick(self):
        """Test that allocate updates stats tick."""
        self.metronome.allocate(tick=5)
        self.assertEqual(self.metronome.stats.tick, 5)

    def test_record_stats_updates_tick(self):
        """Test that record_stats updates tick."""
        self.metronome.record_stats(tick=10)
        self.assertEqual(self.metronome.stats.tick, 10)

    def test_different_grid_sizes(self):
        """Test allocation with different grid sizes."""
        cfg_small = EngineConfig(grid_w=8, grid_h=8, microtick_cap_per_region=5)
        metronome_small = Metronome(cfg_small)
        budgets = metronome_small.allocate(tick=1)
        # 8 * 8 * 5 = 320
        self.assertEqual(budgets["quanta_micro_ops"], 320)

    def test_different_microtick_cap(self):
        """Test allocation with different microtick cap."""
        cfg = EngineConfig(grid_w=10, grid_h=10, microtick_cap_per_region=20)
        metronome = Metronome(cfg)
        budgets = metronome.allocate(tick=1)
        # 10 * 10 * 20 = 2000
        self.assertEqual(budgets["quanta_micro_ops"], 2000)

    def test_sequential_allocations(self):
        """Test that sequential allocations update tick correctly."""
        for tick in range(1, 6):
            self.metronome.allocate(tick=tick)
            self.assertEqual(self.metronome.stats.tick, tick)


class TestMetronomeBudgetScaling(unittest.TestCase):
    """Tests for Metronome budget scaling behavior."""

    def test_budget_scales_with_grid(self):
        """Test that budget scales linearly with grid size."""
        cfg1 = EngineConfig(grid_w=10, grid_h=10, microtick_cap_per_region=1)
        cfg2 = EngineConfig(grid_w=20, grid_h=20, microtick_cap_per_region=1)
        m1 = Metronome(cfg1)
        m2 = Metronome(cfg2)
        b1 = m1.allocate(1)["quanta_micro_ops"]
        b2 = m2.allocate(1)["quanta_micro_ops"]
        # 20*20 = 4 * 10*10
        self.assertEqual(b2, 4 * b1)

    def test_budget_scales_with_microtick_cap(self):
        """Test that budget scales linearly with microtick cap."""
        cfg1 = EngineConfig(grid_w=10, grid_h=10, microtick_cap_per_region=5)
        cfg2 = EngineConfig(grid_w=10, grid_h=10, microtick_cap_per_region=10)
        m1 = Metronome(cfg1)
        m2 = Metronome(cfg2)
        b1 = m1.allocate(1)["quanta_micro_ops"]
        b2 = m2.allocate(1)["quanta_micro_ops"]
        self.assertEqual(b2, 2 * b1)


if __name__ == "__main__":
    unittest.main()
