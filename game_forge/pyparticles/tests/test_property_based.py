"""
Eidosian PyParticles V6.2 - Property-Based Tests

Uses Hypothesis to verify physical invariants hold across
arbitrary inputs and edge cases.
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import pytest

from pyparticles.physics.exclusion.kernels import (
    compute_exclusion_force_wave, compute_spin_coupling_torque
)


# Strategies for generating valid physics values
positive_float = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
distance = st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
angle = st.floats(min_value=0.0, max_value=6.28, allow_nan=False, allow_infinity=False)
wave_amp = st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False)
spin_state = st.sampled_from([-1, 0, 1])
behavior = st.sampled_from([0, 1, 2])  # classical, fermionic, bosonic


class TestExclusionInvariants:
    """Property tests for exclusion mechanics."""
    
    @given(
        dist=distance,
        r_i=positive_float, r_j=positive_float,
        spin_i=spin_state, spin_j=spin_state,
        behavior=behavior
    )
    @settings(max_examples=100, deadline=500)
    def test_exclusion_force_finite(self, dist, r_i, r_j, spin_i, spin_j, behavior):
        """Exclusion force should always be finite."""
        force, torque = compute_exclusion_force_wave(
            dist, r_i, r_j, 
            1.0, 1.0, 0.0, 0.0,  # circular (no wave)
            0.0, 0.0,  # angles
            spin_i, spin_j, behavior,
            exclusion_strength=5.0, exclusion_radius_factor=2.0
        )
        assert np.isfinite(force)
        assert np.isfinite(torque)
    
    @given(dist=distance, r_i=positive_float, r_j=positive_float)
    @settings(max_examples=100, deadline=500)
    def test_classical_soft_sphere(self, dist, r_i, r_j):
        """Classical particles should only repel when overlapping."""
        force, _ = compute_exclusion_force_wave(
            dist, r_i, r_j,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0,
            1, -1, 0,  # classical behavior
            exclusion_strength=10.0, exclusion_radius_factor=3.0
        )
        # Classical: repels when gap < 0, else no force
        if dist >= r_i + r_j:
            assert force == 0.0
    
    @given(
        dist=distance,
        spin_i=spin_state, spin_j=spin_state,
        ang_vel_i=st.floats(min_value=-5, max_value=5, allow_nan=False),
        ang_vel_j=st.floats(min_value=-5, max_value=5, allow_nan=False)
    )
    @settings(max_examples=100, deadline=500)
    def test_spin_coupling_bounded(self, dist, spin_i, spin_j, ang_vel_i, ang_vel_j):
        """Spin coupling torque should be bounded."""
        result = compute_spin_coupling_torque(
            spin_i, spin_j, ang_vel_i, ang_vel_j, dist,
            coupling_strength=1.0, interaction_range=10.0
        )
        assert np.isfinite(result)


class TestEnergyConservation:
    """Tests for energy conservation properties."""
    
    @given(
        n_particles=st.integers(min_value=10, max_value=100),
        n_steps=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=15, deadline=15000)
    def test_energy_bounded(self, n_particles, n_steps):
        """Total energy should remain bounded over time."""
        from pyparticles.core.types import SimulationConfig
        from pyparticles.physics.engine import PhysicsEngine
        
        cfg = SimulationConfig(
            num_particles=n_particles,
            num_types=4,
            world_size=20.0,
            max_velocity=5.0,
            thermostat_enabled=True,
            target_temperature=1.0
        )
        engine = PhysicsEngine(cfg)
        
        # Compute initial kinetic energy
        def kinetic_energy(engine):
            v = engine.state.vel[:engine.state.active]
            return 0.5 * np.sum(v * v)
        
        # Run simulation
        for _ in range(n_steps):
            engine.update()
        
        final_ke = kinetic_energy(engine)
        
        # With thermostat and velocity cap, energy should be bounded
        max_possible = 0.5 * n_particles * cfg.max_velocity ** 2
        assert final_ke <= max_possible * 1.1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_particles(self):
        """Engine should handle zero active particles."""
        from pyparticles.core.types import SimulationConfig
        from pyparticles.physics.engine import PhysicsEngine
        
        cfg = SimulationConfig(num_particles=100)
        engine = PhysicsEngine(cfg)
        engine.set_active_count(0)
        
        # Should not crash
        engine.update()
        assert engine.state.active == 0
    
    def test_single_particle(self):
        """Engine should handle single particle."""
        from pyparticles.core.types import SimulationConfig
        from pyparticles.physics.engine import PhysicsEngine
        
        cfg = SimulationConfig(num_particles=100)
        engine = PhysicsEngine(cfg)
        engine.set_active_count(1)
        
        # Should not crash
        engine.update()
        assert engine.state.active == 1
    
    @given(n_types=st.integers(min_value=1, max_value=32))
    @settings(max_examples=15, deadline=10000)
    def test_arbitrary_species_count(self, n_types):
        """Engine should handle arbitrary species counts."""
        from pyparticles.core.types import SimulationConfig
        from pyparticles.physics.engine import PhysicsEngine
        
        cfg = SimulationConfig(num_particles=100, num_types=n_types)
        engine = PhysicsEngine(cfg)
        
        engine.update()
        assert engine.cfg.num_types == n_types
    
    def test_spin_statistics(self):
        """Spin statistics should sum correctly."""
        from pyparticles.core.types import SimulationConfig
        from pyparticles.physics.engine import PhysicsEngine
        
        cfg = SimulationConfig(num_particles=500, num_types=4)
        engine = PhysicsEngine(cfg)
        
        # Run a few steps
        for _ in range(5):
            engine.update()
        
        stats = engine.get_spin_statistics()
        n_up, n_down, n_none, total_spin, correlation = stats
        
        # Spins should sum to active count
        assert n_up + n_down + n_none == cfg.num_particles
        # Total spin is n_up - n_down
        assert total_spin == n_up - n_down


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
