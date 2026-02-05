"""
Eidosian PyParticles V6 - Wave Mechanics Module

Advanced wave-particle interactions including:
- Interference patterns (constructive/destructive)
- Dynamic crest/trough/zero-crossing detection
- Phase-dependent force modulation
- Standing wave formation
- Multi-frequency superposition
"""

from .types import (
    WaveMode, WaveFeature, WaveInterference,
    WaveConfig, WaveState, WaveInteraction
)
from .kernels import (
    compute_wave_height, compute_wave_derivative,
    detect_wave_feature, compute_interference,
    compute_wave_force, compute_wave_torque,
    update_wave_phases, compute_wave_energy,
    detect_standing_wave_pairs
)
from .analyzer import WaveAnalyzer, WaveStatistics
from .registry import WaveProfile, WaveRegistry, create_wave_preset

__all__ = [
    # Types
    'WaveMode', 'WaveFeature', 'WaveInterference',
    'WaveConfig', 'WaveState', 'WaveInteraction',
    # Kernels
    'compute_wave_height', 'compute_wave_derivative',
    'detect_wave_feature', 'compute_interference',
    'compute_wave_force', 'compute_wave_torque',
    'update_wave_phases', 'compute_wave_energy',
    'detect_standing_wave_pairs',
    # Analysis
    'WaveAnalyzer', 'WaveStatistics',
    # Registry
    'WaveProfile', 'WaveRegistry', 'create_wave_preset',
]
