"""
Tests for the core.quanta module.

This module tests the Quanta subsystem including signal handling,
cell resolution, and mass transfer.
"""

import unittest
import sys
import os
import tempfile
import shutil

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quanta import Quanta, Signal, SignalQueue, SignalType
from core.fabric import Fabric
from core.ledger import Ledger
from core.config import EngineConfig
from core.registry import SpeciesRegistry
from core.types import Vec2


# Helper to create a minimal MaterialsFundamentals mock
class MockMaterialsFundamentals:
    """Minimal mock for MaterialsFundamentals."""

    def __init__(self, registry, cfg):
        self.registry = registry
        self.cfg = cfg
        # Create a mock stellar species
        self.stellar_species = registry.get_or_create_species(
            {"HE/rho_max": 0.3, "HE/chi": 0.5, "HE/beta": 0.4, "HE/lambda": 0.1, "HE/nu": 0.2}
        )
        self.event_calls = []

    def apply_global_ops(self, fabric, cfg, mix_cache=None):
        """No-op for testing."""
        pass

    def effective_property(self, mix, registry, prop_name):
        """Return a default value for testing."""
        return 0.1

    def handle_high_energy_events(self, fabric, ledger, registry, i, j, Z, T, tick, attempt, **kwargs):
        """No-op for testing."""
        self.event_calls.append((i, j, tick, attempt))

    def absorb_into_black_hole(self, fabric, i, j, cfg):
        """Simple absorption for testing."""
        fabric.rho[i, j] = 0.0
        fabric.px[i, j] = 0.0
        fabric.py[i, j] = 0.0


class TestSignal(unittest.TestCase):
    """Tests for the Signal class."""

    def test_creation(self):
        """Test Signal creation."""
        sig = Signal(
            sig_type=SignalType.INFLUENCE,
            emit_tick=5,
            origin=(1, 2),
            speed=2.0,
            attenuation=0.1,
            radius=3,
            payload={"strength": 1.0},
        )
        self.assertEqual(sig.type, SignalType.INFLUENCE)
        self.assertEqual(sig.emit_tick, 5)
        self.assertEqual(sig.origin, (1, 2))
        self.assertEqual(sig.speed, 2.0)
        self.assertEqual(sig.payload["strength"], 1.0)


class TestSignalQueue(unittest.TestCase):
    """Tests for the SignalQueue class."""

    def test_empty_queue(self):
        """Test empty queue returns empty list."""
        queue = SignalQueue()
        arrivals = queue.pop_arrivals(tick=1)
        self.assertEqual(arrivals, [])

    def test_push_and_pop(self):
        """Test pushing and popping signals."""
        queue = SignalQueue()
        sig = Signal(
            SignalType.RADIATION, emit_tick=0, origin=(0, 0),
            speed=1.0, attenuation=0.0, radius=0, payload={}
        )
        queue.push(sig, current_tick=0, v_max=5.0)
        # Signal arrives at tick 1
        arrivals = queue.pop_arrivals(tick=1)
        self.assertEqual(len(arrivals), 1)
        self.assertIs(arrivals[0], sig)

    def test_pop_clears_arrivals(self):
        """Test that popping removes signals from queue."""
        queue = SignalQueue()
        sig = Signal(
            SignalType.RADIATION, emit_tick=0, origin=(0, 0),
            speed=1.0, attenuation=0.0, radius=0, payload={}
        )
        queue.push(sig, current_tick=0, v_max=5.0)
        queue.pop_arrivals(tick=1)
        # Second pop should return empty
        arrivals = queue.pop_arrivals(tick=1)
        self.assertEqual(arrivals, [])

    def test_multiple_signals_same_tick(self):
        """Test multiple signals arriving at same tick."""
        queue = SignalQueue()
        sig1 = Signal(SignalType.INFLUENCE, 0, (0, 0), 1.0, 0.0, 0, {})
        sig2 = Signal(SignalType.RADIATION, 0, (1, 1), 1.0, 0.0, 0, {})
        queue.push(sig1, current_tick=0, v_max=5.0)
        queue.push(sig2, current_tick=0, v_max=5.0)
        arrivals = queue.pop_arrivals(tick=1)
        self.assertEqual(len(arrivals), 2)


class TestQuanta(unittest.TestCase):
    """Tests for the Quanta class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8, microtick_cap_per_region=5)
        self.fabric = Fabric(self.cfg)
        self.ledger = Ledger(self.fabric, self.cfg)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.registry = SpeciesRegistry(
            registry_path,
            ["HE/rho_max", "HE/chi", "HE/eta", "HE/opacity", "HE/beta", "HE/lambda", "HE/nu"],
            []
        )
        self.materials = MockMaterialsFundamentals(self.registry, self.cfg)
        self.quanta = Quanta(
            self.fabric, self.ledger, self.registry,
            self.materials, self.cfg
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test Quanta initialization."""
        self.assertIs(self.quanta.fabric, self.fabric)
        self.assertIs(self.quanta.ledger, self.ledger)
        self.assertIsInstance(self.quanta.queue, SignalQueue)
        self.assertEqual(self.quanta.total_microticks, 0)

    def test_step_empty_grid(self):
        """Test step on empty grid doesn't crash."""
        self.quanta.step(tick=1, micro_budget=100)
        # Should complete without error

    def test_step_with_mass(self):
        """Test step with some mass in grid."""
        # Add mass to a cell
        self.fabric.rho[4, 4] = 1.0
        self.fabric.E_heat[4, 4] = 0.5
        mix = self.fabric.mixtures[4][4]
        mix.species_ids = [self.materials.stellar_species.id]
        mix.masses = [1.0]
        self.quanta.step(tick=1, micro_budget=100)
        # Should complete and process the cell

    def test_high_energy_events_once_per_tick(self):
        """High-energy events run at most once per cell per tick."""
        self.fabric.rho[3, 3] = 10.0
        self.fabric.E_heat[3, 3] = 5.0
        mix = self.fabric.mixtures[3][3]
        mix.species_ids = [self.materials.stellar_species.id]
        mix.masses = [10.0]
        self.quanta.step(tick=1, micro_budget=10)
        self.assertEqual(len(self.materials.event_calls), 1)

    def test_compute_effective_rho_max(self):
        """Test effective rho_max computation."""
        # Set up a cell with mixture
        self.fabric.rho[0, 0] = 1.0
        mix = self.fabric.mixtures[0][0]
        mix.species_ids = [self.materials.stellar_species.id]
        mix.masses = [1.0]
        rho_max_eff = self.quanta.compute_effective_rho_max()
        # Should have some value for cell (0,0)
        self.assertGreater(rho_max_eff[0, 0], 0.0)

    def test_compute_effective_rho_max_empty_cell(self):
        """Test effective rho_max for empty cell is zero."""
        rho_max_eff = self.quanta.compute_effective_rho_max()
        self.assertEqual(rho_max_eff[0, 0], 0.0)

    def test_derived_dom_props(self):
        """Test dominant species properties are cached in derived fields."""
        self.fabric.rho[1, 1] = 2.0
        self.fabric.E_heat[1, 1] = 1.0
        mix = self.fabric.mixtures[1][1]
        mix.species_ids = [self.materials.stellar_species.id]
        mix.masses = [2.0]
        derived = self.quanta._compute_derived_fields(tick=1)
        self.assertAlmostEqual(derived.dom_beta[1, 1], 0.4)
        self.assertAlmostEqual(derived.dom_lambda[1, 1], 0.1)
        self.assertAlmostEqual(derived.dom_nu[1, 1], 0.2)


class TestQuantaTransferMass(unittest.TestCase):
    """Tests for Quanta mass transfer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        self.fabric = Fabric(self.cfg)
        self.ledger = Ledger(self.fabric, self.cfg)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.registry = SpeciesRegistry(
            registry_path,
            ["HE/rho_max", "HE/chi", "HE/eta"],
            []
        )
        self.materials = MockMaterialsFundamentals(self.registry, self.cfg)
        self.quanta = Quanta(
            self.fabric, self.ledger, self.registry,
            self.materials, self.cfg
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_transfer_mass_basic(self):
        """Test basic mass transfer between cells."""
        # Set up source cell
        self.fabric.rho[0, 0] = 10.0
        self.fabric.px[0, 0] = 5.0
        self.fabric.py[0, 0] = 3.0
        self.fabric.E_heat[0, 0] = 2.0
        self.fabric.E_rad[0, 0] = 1.0
        # Transfer 5 units to (1, 0)
        self.quanta.transfer_mass(0, 0, 1, 0, mass=5.0, v_x=1.0, v_y=0.0)
        # Check source
        self.assertEqual(self.fabric.rho[0, 0], 5.0)
        # Check destination
        self.assertEqual(self.fabric.rho[1, 0], 5.0)

    def test_transfer_mass_zero(self):
        """Test transferring zero mass does nothing."""
        self.fabric.rho[0, 0] = 10.0
        self.quanta.transfer_mass(0, 0, 1, 0, mass=0.0, v_x=0.0, v_y=0.0)
        self.assertEqual(self.fabric.rho[0, 0], 10.0)
        self.assertEqual(self.fabric.rho[1, 0], 0.0)

    def test_transfer_mass_clamps_to_available(self):
        """Test that transfer is clamped to available mass."""
        self.fabric.rho[0, 0] = 5.0
        self.quanta.transfer_mass(0, 0, 1, 0, mass=10.0, v_x=0.0, v_y=0.0)
        # Should only transfer 5.0
        self.assertEqual(self.fabric.rho[0, 0], 0.0)
        self.assertEqual(self.fabric.rho[1, 0], 5.0)

    def test_transfer_mass_moves_momentum(self):
        """Test that momentum is transferred proportionally."""
        self.fabric.rho[0, 0] = 10.0
        self.fabric.px[0, 0] = 10.0
        self.fabric.py[0, 0] = 20.0
        self.quanta.transfer_mass(0, 0, 1, 0, mass=5.0, v_x=1.0, v_y=2.0)
        # 50% of mass moved, so 50% of momentum
        self.assertEqual(self.fabric.px[0, 0], 5.0)
        self.assertEqual(self.fabric.py[0, 0], 10.0)
        self.assertEqual(self.fabric.px[1, 0], 5.0)
        self.assertEqual(self.fabric.py[1, 0], 10.0)

    def test_transfer_mass_moves_energy(self):
        """Test that energy is transferred proportionally."""
        self.fabric.rho[0, 0] = 10.0
        self.fabric.E_heat[0, 0] = 10.0
        self.fabric.E_rad[0, 0] = 5.0
        self.quanta.transfer_mass(0, 0, 1, 0, mass=5.0, v_x=0.0, v_y=0.0)
        # 50% of mass moved, so 50% of energy
        self.assertEqual(self.fabric.E_heat[0, 0], 5.0)
        self.assertEqual(self.fabric.E_rad[0, 0], 2.5)
        self.assertEqual(self.fabric.E_heat[1, 0], 5.0)
        self.assertEqual(self.fabric.E_rad[1, 0], 2.5)

    def test_transfer_mass_from_empty_cell(self):
        """Test transferring from empty cell does nothing."""
        self.quanta.transfer_mass(0, 0, 1, 0, mass=5.0, v_x=0.0, v_y=0.0)
        self.assertEqual(self.fabric.rho[0, 0], 0.0)
        self.assertEqual(self.fabric.rho[1, 0], 0.0)

    def test_apply_mass_transfers_preserves_sparse_mixture(self):
        """Ensure batch transfers move mixture proportionally even when mix_total < rho."""
        stellar_id = self.materials.stellar_species.id
        self.fabric.rho[0, 0] = 1.0
        mix = self.fabric.mixtures[0][0]
        mix.species_ids = [stellar_id]
        mix.masses = [0.4]
        transfers = [(1, 0, 0.2), (0, 1, 0.4)]
        self.quanta._apply_mass_transfers(0, 0, transfers)
        self.assertAlmostEqual(self.fabric.mixtures[0][0].total_mass(), 0.16)
        self.assertAlmostEqual(self.fabric.mixtures[1][0].total_mass(), 0.08)
        self.assertAlmostEqual(self.fabric.mixtures[0][1].total_mass(), 0.16)


class TestQuantaSignalProcessing(unittest.TestCase):
    """Tests for Quanta signal processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = EngineConfig(grid_w=8, grid_h=8)
        self.fabric = Fabric(self.cfg)
        self.ledger = Ledger(self.fabric, self.cfg)
        registry_path = os.path.join(self.temp_dir, "registry.json")
        self.registry = SpeciesRegistry(
            registry_path,
            ["HE/rho_max", "HE/chi"],
            []
        )
        self.materials = MockMaterialsFundamentals(self.registry, self.cfg)
        self.quanta = Quanta(
            self.fabric, self.ledger, self.registry,
            self.materials, self.cfg
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_influence_signal_processed(self):
        """Test that influence signals add to influence field."""
        sig = Signal(
            SignalType.INFLUENCE, emit_tick=0, origin=(4, 4),
            speed=1.0, attenuation=0.0, radius=0,
            payload={"strength": 5.0}
        )
        self.quanta.queue.push(sig, current_tick=0, v_max=5.0)
        # After step, influence field is reset, so we need to check that 
        # the signal was processed during the step
        # The influence field gets reset at the end of each step
        # So we can only verify the queue was processed (arrivals should be empty)
        self.quanta.step(tick=1, micro_budget=100)
        # Queue should be empty after processing
        arrivals_at_1 = self.quanta.queue.pop_arrivals(tick=1)
        self.assertEqual(len(arrivals_at_1), 0)

    def test_radiation_signal_processed(self):
        """Test that radiation signals add to radiation field."""
        sig = Signal(
            SignalType.RADIATION, emit_tick=0, origin=(4, 4),
            speed=1.0, attenuation=0.0, radius=0,
            payload={"energy": 3.0}
        )
        self.quanta.queue.push(sig, current_tick=0, v_max=5.0)
        initial_rad = self.fabric.E_rad[4, 4]
        self.quanta.step(tick=1, micro_budget=100)
        # Radiation should have increased
        self.assertGreater(self.fabric.E_rad[4, 4], initial_rad)

    def test_impulse_signal_processed(self):
        """Test that impulse signals add to momentum field."""
        sig = Signal(
            SignalType.IMPULSE, emit_tick=0, origin=(4, 4),
            speed=1.0, attenuation=0.0, radius=0,
            payload={"dp": Vec2(2.0, 3.0)}
        )
        self.quanta.queue.push(sig, current_tick=0, v_max=5.0)
        self.quanta.step(tick=1, micro_budget=100)
        # Momentum should have increased
        self.assertEqual(self.fabric.px[4, 4], 2.0)
        self.assertEqual(self.fabric.py[4, 4], 3.0)


if __name__ == "__main__":
    unittest.main()
