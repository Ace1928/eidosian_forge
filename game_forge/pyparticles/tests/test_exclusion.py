"""
Tests for Exclusion Mechanics Module (Phase 4).
"""

import pytest
import numpy as np
from pyparticles.physics.exclusion.types import (
    SpinState, ParticleBehavior, SpinConfig, ExclusionConfig, SpinStatistics
)
from pyparticles.physics.exclusion.kernels import (
    compute_exclusion_force, compute_spin_interaction, 
    compute_spin_statistics, initialize_spins
)
from pyparticles.physics.exclusion.registry import ExclusionRegistry, ExclusionPreset


class TestSpinState:
    """Tests for SpinState enum."""
    
    def test_spin_values(self):
        assert SpinState.DOWN == -1
        assert SpinState.NONE == 0
        assert SpinState.UP == 1
    
    def test_spin_multiplication(self):
        """Same spin product should be positive, opposite negative."""
        assert SpinState.UP * SpinState.UP == 1
        assert SpinState.DOWN * SpinState.DOWN == 1
        assert SpinState.UP * SpinState.DOWN == -1


class TestParticleBehavior:
    """Tests for ParticleBehavior enum."""
    
    def test_behavior_values(self):
        assert ParticleBehavior.CLASSICAL == 0
        assert ParticleBehavior.FERMIONIC == 1
        assert ParticleBehavior.BOSONIC == 2


class TestSpinConfig:
    """Tests for SpinConfig dataclass."""
    
    def test_default_config(self):
        cfg = SpinConfig.default(8)
        assert len(cfg.spin_enabled) == 8
        assert all(cfg.spin_enabled)
        assert len(cfg.flip_threshold) == 8
        assert len(cfg.flip_probability) == 8
        assert len(cfg.coupling_strength) == 8
    
    def test_mixed_config(self):
        cfg = SpinConfig.mixed(10)
        assert len(cfg.spin_enabled) == 10
        # First half should have spin enabled
        assert cfg.spin_enabled[0] == True
        assert cfg.spin_enabled[4] == True
        # Second half should be spinless
        assert cfg.spin_enabled[5] == False


class TestExclusionConfig:
    """Tests for ExclusionConfig dataclass."""
    
    def test_default_values(self):
        cfg = ExclusionConfig()
        assert cfg.exclusion_strength == 20.0
        assert cfg.exclusion_radius_factor == 2.0
        assert cfg.allow_spin_flips == True
    
    def test_initialize(self):
        cfg = ExclusionConfig()
        cfg.initialize(6)
        assert cfg.type_behavior.shape == (6,)
        assert cfg.behavior_matrix.shape == (6, 6)
    
    def test_all_fermionic(self):
        cfg = ExclusionConfig.all_fermionic(4, strength=30.0)
        assert cfg.exclusion_strength == 30.0
        assert all(cfg.type_behavior == ParticleBehavior.FERMIONIC)
        assert np.all(cfg.behavior_matrix == ParticleBehavior.FERMIONIC)
    
    def test_all_bosonic(self):
        cfg = ExclusionConfig.all_bosonic(4)
        assert cfg.exclusion_strength == 0.0
        assert all(cfg.type_behavior == ParticleBehavior.BOSONIC)


class TestExclusionForce:
    """Tests for exclusion force kernel."""
    
    def test_no_force_outside_range(self):
        force = compute_exclusion_force(
            dist=2.0, r_i=0.1, r_j=0.1,
            spin_i=1, spin_j=1,
            behavior=ParticleBehavior.FERMIONIC,
            exclusion_strength=20.0,
            exclusion_radius_factor=2.0
        )
        assert force == 0.0
    
    def test_classical_no_force(self):
        """Classical particles have no exclusion force."""
        force = compute_exclusion_force(
            dist=0.1, r_i=0.1, r_j=0.1,
            spin_i=1, spin_j=1,
            behavior=ParticleBehavior.CLASSICAL,
            exclusion_strength=20.0,
            exclusion_radius_factor=2.0
        )
        assert force == 0.0
    
    def test_fermionic_same_spin_strong(self):
        """Same spin fermions should have strong repulsion."""
        force = compute_exclusion_force(
            dist=0.15, r_i=0.1, r_j=0.1,
            spin_i=1, spin_j=1,  # Same spin
            behavior=ParticleBehavior.FERMIONIC,
            exclusion_strength=20.0,
            exclusion_radius_factor=2.0
        )
        assert force > 0  # Repulsion
    
    def test_fermionic_opposite_spin_weaker(self):
        """Opposite spin fermions should have weaker repulsion (pairing)."""
        force_same = compute_exclusion_force(
            dist=0.15, r_i=0.1, r_j=0.1,
            spin_i=1, spin_j=1,  # Same
            behavior=ParticleBehavior.FERMIONIC,
            exclusion_strength=20.0,
            exclusion_radius_factor=2.0
        )
        force_opp = compute_exclusion_force(
            dist=0.15, r_i=0.1, r_j=0.1,
            spin_i=1, spin_j=-1,  # Opposite
            behavior=ParticleBehavior.FERMIONIC,
            exclusion_strength=20.0,
            exclusion_radius_factor=2.0
        )
        assert force_same > force_opp
    
    def test_bosonic_attraction(self):
        """Bosons should slightly attract at close range."""
        force = compute_exclusion_force(
            dist=0.1, r_i=0.1, r_j=0.1,
            spin_i=0, spin_j=0,
            behavior=ParticleBehavior.BOSONIC,
            exclusion_strength=20.0,
            exclusion_radius_factor=2.0
        )
        assert force < 0  # Attraction


class TestSpinInteraction:
    """Tests for spin-spin interaction."""
    
    def test_no_interaction_spinless(self):
        """Spinless particles should have no spin interaction."""
        mod = compute_spin_interaction(
            spin_i=0, spin_j=1,
            dist=0.5,
            coupling_i=1.0, coupling_j=1.0,
            interaction_range=1.0
        )
        assert mod == 0.0
    
    def test_aligned_positive(self):
        """Aligned spins should give positive modifier."""
        mod = compute_spin_interaction(
            spin_i=1, spin_j=1,
            dist=0.5,
            coupling_i=1.0, coupling_j=1.0,
            interaction_range=1.0
        )
        assert mod > 0
    
    def test_antialigned_negative(self):
        """Anti-aligned spins should give negative modifier."""
        mod = compute_spin_interaction(
            spin_i=1, spin_j=-1,
            dist=0.5,
            coupling_i=1.0, coupling_j=1.0,
            interaction_range=1.0
        )
        assert mod < 0
    
    def test_distance_falloff(self):
        """Interaction should decrease with distance."""
        mod_near = compute_spin_interaction(
            spin_i=1, spin_j=1,
            dist=0.2,
            coupling_i=1.0, coupling_j=1.0,
            interaction_range=1.0
        )
        mod_far = compute_spin_interaction(
            spin_i=1, spin_j=1,
            dist=0.8,
            coupling_i=1.0, coupling_j=1.0,
            interaction_range=1.0
        )
        assert mod_near > mod_far


class TestSpinStatistics:
    """Tests for spin statistics computation."""
    
    def test_compute_statistics(self):
        spin = np.array([1, 1, -1, 0, 1, -1], dtype=np.int8)
        pos = np.random.uniform(-1, 1, (6, 2)).astype(np.float32)
        
        n_up, n_down, n_none, total, corr = compute_spin_statistics(
            spin, pos, 6, correlation_range=5.0
        )
        
        assert n_up == 3
        assert n_down == 2
        assert n_none == 1
        assert total == 1  # 3 - 2


class TestInitializeSpins:
    """Tests for spin initialization."""
    
    def test_initialize_random(self):
        spin = np.zeros(100, dtype=np.int8)
        colors = np.random.randint(0, 4, 100, dtype=np.int32)
        spin_enabled = np.array([True, True, False, False], dtype=np.bool_)
        
        initialize_spins(spin, colors, spin_enabled, 100, seed=42)
        
        # Check that spin-enabled types have spin
        for i in range(100):
            t = colors[i]
            if spin_enabled[t]:
                assert spin[i] in [-1, 1]
            else:
                assert spin[i] == 0


class TestExclusionRegistry:
    """Tests for ExclusionRegistry."""
    
    def test_create_registry(self):
        reg = ExclusionRegistry(8)
        assert reg.n_types == 8
        assert len(reg.presets) >= 4
    
    def test_presets_available(self):
        reg = ExclusionRegistry(4)
        assert 'fermi_gas' in reg.presets
        assert 'bose_einstein' in reg.presets
        assert 'electron_gas' in reg.presets
        assert 'mixed' in reg.presets
        assert 'classical' in reg.presets
    
    def test_apply_preset(self):
        reg = ExclusionRegistry(6)
        reg.apply_preset('fermi_gas')
        assert reg.config.exclusion_strength == 25.0
        assert all(reg.config.type_behavior == ParticleBehavior.FERMIONIC)
    
    def test_pack_for_kernel(self):
        reg = ExclusionRegistry(4)
        packed = reg.pack_for_kernel()
        
        assert 'behavior_matrix' in packed
        assert 'exclusion_strength' in packed
        assert 'spin_enabled' in packed
        assert packed['behavior_matrix'].shape == (4, 4)
    
    def test_serialization(self):
        reg = ExclusionRegistry(6)
        reg.apply_preset('electron_gas')
        
        data = reg.to_dict()
        reg2 = ExclusionRegistry.from_dict(data)
        
        assert reg2.n_types == 6
        assert reg2.config.exclusion_strength == reg.config.exclusion_strength
    
    def test_resize(self):
        reg = ExclusionRegistry(4)
        reg.resize(8)
        assert reg.n_types == 8
        assert reg.config.behavior_matrix.shape == (8, 8)


class TestSpinStatisticsDataclass:
    """Tests for SpinStatistics dataclass."""
    
    def test_repr(self):
        stats = SpinStatistics(n_up=10, n_down=5, n_none=2, total_spin=5)
        s = repr(stats)
        assert "up=10" in s
        assert "down=5" in s
        assert "net=5" in s
