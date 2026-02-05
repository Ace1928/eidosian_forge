"""
Eidosian PyParticles V6 - Wave Analyzer

Real-time analysis of wave patterns, interference, and emergent structures.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .types import WaveConfig, WaveState, WaveFeature, WaveInterference
from .kernels import (
    compute_wave_height, compute_wave_derivative, detect_wave_feature,
    compute_interference, detect_standing_wave_pairs, compute_wave_energy
)


@dataclass
class WaveStatistics:
    """Statistical summary of wave activity."""
    # Counts
    n_active_waves: int = 0
    n_standing_pairs: int = 0
    n_constructive: int = 0
    n_destructive: int = 0
    
    # Averages
    mean_wave_energy: float = 0.0
    mean_phase_velocity: float = 0.0
    
    # Extremes
    max_wave_energy: float = 0.0
    max_amplitude: float = 0.0
    
    # Phase distribution
    phase_coherence: float = 0.0  # 0 = random, 1 = all aligned


@dataclass
class WaveAnalyzer:
    """
    Analyzes wave patterns and emergent structures in the simulation.
    
    Provides real-time statistics and pattern detection for:
    - Wave activity levels
    - Standing wave formation
    - Interference patterns
    - Phase coherence
    """
    
    config: WaveConfig = field(default_factory=WaveConfig)
    
    # History for trend analysis
    history_length: int = 100
    energy_history: List[float] = field(default_factory=list)
    standing_history: List[int] = field(default_factory=list)
    
    def analyze(self, 
                wave_state: WaveState,
                pos: np.ndarray,
                angle: np.ndarray,
                colors: np.ndarray,
                species_freq: np.ndarray,
                species_amp: np.ndarray,
                n_active: int) -> WaveStatistics:
        """
        Perform full wave analysis.
        
        Args:
            wave_state: Current wave state
            pos: (N, 2) particle positions
            angle: (N,) particle angles
            colors: (N,) particle type indices
            species_freq: (T,) wave frequencies per species
            species_amp: (T,) wave amplitudes per species
            n_active: Number of active particles
            
        Returns:
            WaveStatistics with current analysis
        """
        stats = WaveStatistics()
        
        if n_active == 0:
            return stats
        
        # Count active waves (non-zero amplitude)
        active_amps = species_amp[colors[:n_active]]
        stats.n_active_waves = np.sum(active_amps > 1e-6)
        
        # Wave energy
        energies = compute_wave_energy(
            wave_state.phase, wave_state.phase_velocity,
            species_amp, colors, n_active
        )
        stats.mean_wave_energy = np.mean(energies)
        stats.max_wave_energy = np.max(energies) if len(energies) > 0 else 0.0
        
        # Phase velocity
        stats.mean_phase_velocity = np.mean(np.abs(wave_state.phase_velocity[:n_active]))
        
        # Max amplitude
        stats.max_amplitude = np.max(active_amps) if len(active_amps) > 0 else 0.0
        
        # Standing wave detection
        if self.config.mode in (2, 3):  # INTERFERENCE or STANDING
            partners = detect_standing_wave_pairs(
                pos, angle, wave_state.phase, colors,
                species_freq, species_amp, n_active,
                self.config.standing_wave_threshold
            )
            stats.n_standing_pairs = np.sum(partners >= 0) // 2
        
        # Phase coherence (circular mean resultant length)
        if n_active > 1:
            phases = wave_state.phase[:n_active]
            cos_sum = np.sum(np.cos(phases))
            sin_sum = np.sum(np.sin(phases))
            stats.phase_coherence = np.sqrt(cos_sum**2 + sin_sum**2) / n_active
        
        # Update history
        self.energy_history.append(stats.mean_wave_energy)
        self.standing_history.append(stats.n_standing_pairs)
        
        if len(self.energy_history) > self.history_length:
            self.energy_history.pop(0)
            self.standing_history.pop(0)
        
        return stats
    
    def get_energy_trend(self) -> float:
        """Get recent energy trend (-1 to +1)."""
        if len(self.energy_history) < 10:
            return 0.0
        
        recent = np.array(self.energy_history[-10:])
        older = np.array(self.energy_history[-20:-10]) if len(self.energy_history) >= 20 else recent
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        if older_mean < 1e-6:
            return 0.0
        
        return (recent_mean - older_mean) / older_mean
    
    def get_standing_wave_trend(self) -> float:
        """Get trend in standing wave formation."""
        if len(self.standing_history) < 10:
            return 0.0
        
        recent = np.mean(self.standing_history[-10:])
        older = np.mean(self.standing_history[-20:-10]) if len(self.standing_history) >= 20 else recent
        
        if older < 1:
            return 0.0
        
        return (recent - older) / older
    
    def find_hot_spots(self, 
                       wave_state: WaveState,
                       pos: np.ndarray,
                       colors: np.ndarray,
                       species_amp: np.ndarray,
                       n_active: int,
                       grid_size: int = 10) -> np.ndarray:
        """
        Find spatial regions with high wave activity.
        
        Returns a grid of wave energy density.
        
        Args:
            wave_state: Current wave state
            pos: (N, 2) particle positions (normalized to [-1, 1])
            colors: (N,) particle types
            species_amp: (T,) amplitudes
            n_active: Number of active particles
            grid_size: Grid resolution
            
        Returns:
            (grid_size, grid_size) energy density grid
        """
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        energies = compute_wave_energy(
            wave_state.phase, wave_state.phase_velocity,
            species_amp, colors, n_active
        )
        
        for i in range(n_active):
            # Map position to grid cell
            gx = int((pos[i, 0] + 1.0) / 2.0 * grid_size)
            gy = int((pos[i, 1] + 1.0) / 2.0 * grid_size)
            
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            
            grid[gy, gx] += energies[i]
            counts[gy, gx] += 1
        
        # Average where there are particles
        mask = counts > 0
        grid[mask] /= counts[mask]
        
        return grid
    
    def classify_interaction(self,
                            pos_i: np.ndarray, pos_j: np.ndarray,
                            angle_i: float, angle_j: float,
                            freq_i: float, freq_j: float,
                            amp_i: float, amp_j: float) -> Tuple[WaveInterference, float]:
        """
        Classify the wave interaction between two particles.
        
        Args:
            pos_i, pos_j: Particle positions
            angle_i, angle_j: Particle angles
            freq_i, freq_j: Wave frequencies
            amp_i, amp_j: Wave amplitudes
            
        Returns:
            (interference_type, force_multiplier)
        """
        # Vector from i to j
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        
        # Angle from i to j
        phi_ij = np.arctan2(dy, dx)
        
        # Local angles at contact points
        theta_i = phi_ij - angle_i
        theta_j = (phi_ij + np.pi) - angle_j
        
        # Wave heights
        h_i = compute_wave_height(theta_i, freq_i, amp_i)
        h_j = compute_wave_height(theta_j, freq_j, amp_j)
        
        # Wave slopes
        slope_i = compute_wave_derivative(theta_i, freq_i, amp_i)
        slope_j = compute_wave_derivative(theta_j, freq_j, amp_j)
        
        # Feature detection
        feature_i = detect_wave_feature(h_i, slope_i, amp_i)
        feature_j = detect_wave_feature(h_j, slope_j, amp_j)
        
        # Interference classification
        interference_type, multiplier = compute_interference(
            feature_i, feature_j, h_i, h_j, amp_i, amp_j
        )
        
        return WaveInterference(interference_type), multiplier
