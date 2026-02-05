"""
Eidosian PyParticles V6 - Wave Mechanics Tests

Comprehensive tests for wave physics, interference patterns, and analysis.
"""

import pytest
import numpy as np
from pyparticles.physics.wave import (
    WaveMode, WaveFeature, WaveInterference,
    WaveConfig, WaveState, WaveInteraction,
    compute_wave_height, compute_wave_derivative,
    detect_wave_feature, compute_interference,
    compute_wave_force, compute_wave_torque,
    update_wave_phases, compute_wave_energy,
    detect_standing_wave_pairs,
    WaveAnalyzer, WaveStatistics,
    WaveProfile, WaveRegistry, create_wave_preset
)


class TestWaveKernels:
    """Test wave computation kernels."""
    
    def test_wave_height_at_crest(self):
        # At theta=0, h = A * cos(0) = A
        h = compute_wave_height(0.0, 4.0, 0.02)
        assert abs(h - 0.02) < 1e-6
    
    def test_wave_height_at_trough(self):
        # At theta=π/f, h = A * cos(π) = -A for f=1
        # For f=4, trough at theta=π/4
        h = compute_wave_height(np.pi/4, 4.0, 0.02)
        assert abs(h - (-0.02)) < 1e-6
    
    def test_wave_height_at_zero(self):
        # At theta=π/8 for f=4, h = A * cos(π/2) = 0
        h = compute_wave_height(np.pi/8, 4.0, 0.02)
        assert abs(h) < 1e-6
    
    def test_wave_derivative_at_crest(self):
        # Slope at crest should be zero
        dh = compute_wave_derivative(0.0, 4.0, 0.02)
        assert abs(dh) < 1e-6
    
    def test_wave_derivative_at_zero_crossing(self):
        # Slope should be maximum at zero crossing
        # dh/dθ = -A*f*sin(f*θ), max when sin(f*θ) = ±1
        dh = compute_wave_derivative(np.pi/8, 4.0, 0.02)
        expected = -0.02 * 4.0 * np.sin(4.0 * np.pi/8)
        assert abs(dh - expected) < 1e-6


class TestFeatureDetection:
    """Test wave feature classification."""
    
    def test_detect_crest(self):
        h = 0.019  # Close to amplitude
        slope = 0.0
        amp = 0.02
        feature = detect_wave_feature(h, slope, amp)
        assert feature == 1  # CREST
    
    def test_detect_trough(self):
        h = -0.019
        slope = 0.0
        amp = 0.02
        feature = detect_wave_feature(h, slope, amp)
        assert feature == -1  # TROUGH
    
    def test_detect_zero_rising(self):
        h = 0.001  # Near zero
        slope = 0.05  # Positive slope
        amp = 0.02
        feature = detect_wave_feature(h, slope, amp)
        assert feature == 2  # ZERO_RISING
    
    def test_detect_zero_falling(self):
        h = -0.001  # Near zero
        slope = -0.05  # Negative slope
        amp = 0.02
        feature = detect_wave_feature(h, slope, amp)
        assert feature == -2  # ZERO_FALLING


class TestInterference:
    """Test interference pattern calculations."""
    
    def test_constructive_peak(self):
        # Both at crest (positive heights)
        itype, mult = compute_interference(1, 1, 0.018, 0.019, 0.02, 0.02)
        assert itype == 1  # CONSTRUCTIVE_PEAK
        assert mult > 1.0
    
    def test_constructive_trough(self):
        # Both at trough (negative heights)
        itype, mult = compute_interference(-1, -1, -0.018, -0.019, 0.02, 0.02)
        assert itype == 2  # CONSTRUCTIVE_TROUGH
        assert mult > 1.0
    
    def test_destructive(self):
        # One crest, one trough
        itype, mult = compute_interference(1, -1, 0.018, -0.019, 0.02, 0.02)
        assert itype == 3  # DESTRUCTIVE
        assert mult < 1.0


class TestWaveForce:
    """Test wave force calculations."""
    
    def test_no_force_when_separated(self):
        # Positive gap = no overlap
        f = compute_wave_force(0.01, 1.0, 30.0, 8.0)
        assert f == 0.0
    
    def test_repulsion_when_overlapping(self):
        # Negative gap = overlap
        f = compute_wave_force(-0.01, 1.0, 30.0, 8.0)
        assert f > 0.0  # Repulsive
    
    def test_multiplier_affects_force(self):
        f1 = compute_wave_force(-0.01, 1.0, 30.0, 8.0)
        f2 = compute_wave_force(-0.01, 2.0, 30.0, 8.0)
        assert abs(f2 - 2.0 * f1) < 0.01


class TestWaveTorque:
    """Test wave torque calculations."""
    
    def test_no_torque_at_flat_surface(self):
        tau = compute_wave_torque(10.0, 0.0, 0.05)
        assert tau == 0.0
    
    def test_torque_from_slope(self):
        tau = compute_wave_torque(10.0, 0.1, 0.05)
        assert tau != 0.0


class TestPhaseUpdate:
    """Test phase evolution."""
    
    def test_phase_advances(self):
        phase = np.zeros(10, dtype=np.float32)
        phase_vel = np.ones(10, dtype=np.float32) * 2.0
        update_wave_phases(phase, phase_vel, 10, 0.1)
        assert np.allclose(phase, 0.2)
    
    def test_phase_wraps(self):
        phase = np.array([6.0], dtype=np.float32)  # Near 2π
        phase_vel = np.array([1.0], dtype=np.float32)
        update_wave_phases(phase, phase_vel, 1, 1.0)
        assert phase[0] < 2 * np.pi  # Should wrap


class TestWaveEnergy:
    """Test wave energy calculations."""
    
    def test_energy_proportional_to_amp_squared(self):
        phase = np.zeros(2, dtype=np.float32)
        phase_vel = np.ones(2, dtype=np.float32)
        species_amp = np.array([0.01, 0.02], dtype=np.float32)
        types = np.array([0, 1], dtype=np.int32)
        
        energy = compute_wave_energy(phase, phase_vel, species_amp, types, 2)
        # E = 0.5 * A² * ω²
        expected = 0.5 * species_amp ** 2 * 1.0 ** 2
        assert np.allclose(energy, expected)


class TestStandingWaveDetection:
    """Test standing wave pair detection."""
    
    def test_finds_phase_locked_pairs(self):
        pos = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
        angle = np.zeros(2, dtype=np.float32)
        phase = np.array([0.0, 0.0], dtype=np.float32)  # Same phase
        colors = np.zeros(2, dtype=np.int32)
        freq = np.array([4.0], dtype=np.float32)
        amp = np.array([0.02], dtype=np.float32)
        
        partners = detect_standing_wave_pairs(
            pos, angle, phase, colors, freq, amp, 2, 0.8, 0.2
        )
        
        # Should find each other as partners
        assert partners[0] == 1 or partners[1] == 0


class TestWaveConfig:
    """Test wave configuration."""
    
    def test_default_config(self):
        cfg = WaveConfig()
        assert cfg.mode == WaveMode.STANDARD
        assert cfg.repulsion_strength > 0
    
    def test_validation(self):
        cfg = WaveConfig(repulsion_strength=-10.0)
        warnings = cfg.validate()
        assert len(warnings) > 0


class TestWaveState:
    """Test wave state management."""
    
    def test_allocate(self):
        state = WaveState.allocate(100)
        assert state.phase.shape == (100,)
        assert state.standing_partner.dtype == np.int32
    
    def test_reset(self):
        state = WaveState.allocate(50)
        speeds = np.array([1.0, 2.0], dtype=np.float32)
        types = np.random.randint(0, 2, 50).astype(np.int32)
        
        state.reset(50, speeds, types)
        assert np.all(state.standing_partner[:50] == -1)


class TestWaveProfile:
    """Test wave profile configuration."""
    
    def test_create_profile(self):
        p = WaveProfile(
            name="Test",
            frequency=5.0,
            amplitude=0.03,
            phase_speed=2.0
        )
        assert p.frequency == 5.0
    
    def test_to_array(self):
        p = WaveProfile("Test", frequency=4.0, amplitude=0.02, phase_speed=1.5)
        arr = p.to_array()
        assert abs(arr[0] - 4.0) < 1e-6
        assert abs(arr[1] - 0.02) < 1e-6
        assert abs(arr[2] - 1.5) < 1e-6
    
    def test_from_array(self):
        arr = np.array([3.0, 0.025, 2.0], dtype=np.float32)
        p = WaveProfile.from_array(arr, "FromArray")
        assert abs(p.frequency - 3.0) < 1e-6
        assert abs(p.amplitude - 0.025) < 1e-6


class TestWaveRegistry:
    """Test wave registry management."""
    
    def test_add_profile(self):
        reg = WaveRegistry()
        idx = reg.add_profile(WaveProfile("Test", 4.0, 0.02, 1.0))
        assert idx == 0
        assert len(reg.profiles) == 1
    
    def test_pack_for_kernel(self):
        reg = WaveRegistry()
        reg.add_profile(WaveProfile("A", 3.0, 0.01, 1.0))
        reg.add_profile(WaveProfile("B", 4.0, 0.02, 2.0))
        
        freqs, amps, speeds, mat = reg.pack_for_kernel()
        assert len(freqs) == 2
        assert abs(freqs[0] - 3.0) < 1e-6
        assert abs(amps[1] - 0.02) < 1e-6
    
    def test_serialization(self):
        reg = WaveRegistry()
        reg.add_profile(WaveProfile("Test", 4.0, 0.02, 1.5))
        
        data = reg.to_dict()
        restored = WaveRegistry.from_dict(data)
        
        assert len(restored.profiles) == 1
        assert restored.profiles[0].frequency == 4.0


class TestWavePresets:
    """Test preset wave configurations."""
    
    def test_calm_preset(self):
        reg = create_wave_preset('calm', n_types=4)
        assert len(reg.profiles) == 4
        # Calm should have low amplitudes
        for p in reg.profiles:
            assert p.amplitude < 0.02
    
    def test_active_preset(self):
        reg = create_wave_preset('active', n_types=4)
        assert reg.config.mode == WaveMode.INTERFERENCE
        # Active should have higher amplitudes
        for p in reg.profiles:
            assert p.amplitude >= 0.03
    
    def test_interference_preset(self):
        reg = create_wave_preset('interference', n_types=6)
        assert reg.config.constructive_multiplier > 1.5
    
    def test_standing_preset(self):
        reg = create_wave_preset('standing', n_types=4)
        assert reg.config.mode == WaveMode.STANDING
        # All should have same frequency
        freqs = [p.frequency for p in reg.profiles]
        assert len(set(freqs)) == 1


class TestWaveAnalyzer:
    """Test wave analysis functionality."""
    
    def test_analyze_returns_statistics(self):
        analyzer = WaveAnalyzer()
        state = WaveState.allocate(50)
        
        # Initialize state
        state.phase[:50] = np.random.uniform(0, 2*np.pi, 50).astype(np.float32)
        state.phase_velocity[:50] = 1.0
        
        pos = np.random.uniform(-1, 1, (50, 2)).astype(np.float32)
        angle = np.zeros(50, dtype=np.float32)
        colors = np.zeros(50, dtype=np.int32)
        freq = np.array([4.0], dtype=np.float32)
        amp = np.array([0.02], dtype=np.float32)
        
        stats = analyzer.analyze(state, pos, angle, colors, freq, amp, 50)
        
        assert isinstance(stats, WaveStatistics)
        assert stats.n_active_waves >= 0
    
    def test_hot_spots(self):
        analyzer = WaveAnalyzer()
        state = WaveState.allocate(100)
        state.phase[:100] = np.random.uniform(0, 2*np.pi, 100).astype(np.float32)
        state.phase_velocity[:100] = 1.0
        
        # Cluster particles in one area
        pos = np.zeros((100, 2), dtype=np.float32)
        pos[:50, 0] = np.random.uniform(-0.3, -0.1, 50)  # Left cluster
        pos[:50, 1] = np.random.uniform(-0.3, -0.1, 50)
        pos[50:, 0] = np.random.uniform(0.1, 0.3, 50)   # Right cluster
        pos[50:, 1] = np.random.uniform(0.1, 0.3, 50)
        
        colors = np.zeros(100, dtype=np.int32)
        amp = np.array([0.02], dtype=np.float32)
        
        grid = analyzer.find_hot_spots(state, pos, colors, amp, 100, grid_size=5)
        
        assert grid.shape == (5, 5)
        # Should have non-zero values where clusters are
        assert np.sum(grid > 0) > 0
