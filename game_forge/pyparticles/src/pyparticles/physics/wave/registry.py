"""
Eidosian PyParticles V6 - Wave Profile Registry

Manages wave configuration presets and species-specific wave behavior.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .types import WaveConfig, WaveMode


@dataclass
class WaveProfile:
    """
    Wave behavior profile for a particle species.
    
    Defines how waves propagate and interact for a specific type.
    """
    name: str
    
    # Base wave parameters
    frequency: float = 3.0      # Number of wave lobes
    amplitude: float = 0.02     # Wave height relative to radius
    phase_speed: float = 1.0    # Rotation speed (rad/s)
    
    # Interaction modifiers
    self_interference: float = 1.0   # How strongly same-type waves interact
    cross_interference: float = 0.5  # How strongly different-type waves interact
    
    # Advanced features
    decay_rate: float = 0.0     # Amplitude decay over time (0 = no decay)
    harmonic_content: float = 0.0  # Higher harmonic presence (0-1)
    chirality: int = 1          # +1 = right-handed, -1 = left-handed, 0 = both
    
    def to_array(self) -> np.ndarray:
        """Convert to array for kernel use."""
        return np.array([
            self.frequency,
            self.amplitude,
            self.phase_speed,
            self.self_interference,
            self.cross_interference,
            self.decay_rate,
            self.harmonic_content,
            float(self.chirality)
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray, name: str = "Unnamed") -> 'WaveProfile':
        """Create from array."""
        return cls(
            name=name,
            frequency=float(arr[0]),
            amplitude=float(arr[1]),
            phase_speed=float(arr[2]),
            self_interference=float(arr[3]) if len(arr) > 3 else 1.0,
            cross_interference=float(arr[4]) if len(arr) > 4 else 0.5,
            decay_rate=float(arr[5]) if len(arr) > 5 else 0.0,
            harmonic_content=float(arr[6]) if len(arr) > 6 else 0.0,
            chirality=int(arr[7]) if len(arr) > 7 else 1,
        )


@dataclass
class WaveRegistry:
    """
    Registry of wave profiles for all species.
    
    Manages wave behavior configuration and provides arrays for kernels.
    """
    profiles: List[WaveProfile] = field(default_factory=list)
    config: WaveConfig = field(default_factory=WaveConfig)
    
    # Precomputed interaction matrix
    _interaction_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Ensure interaction matrix is computed."""
        if self.profiles and self._interaction_matrix is None:
            self._rebuild_interaction_matrix()
    
    def add_profile(self, profile: WaveProfile) -> int:
        """Add a wave profile and return its index."""
        self.profiles.append(profile)
        self._interaction_matrix = None  # Invalidate
        return len(self.profiles) - 1
    
    def get_profile(self, index: int) -> Optional[WaveProfile]:
        """Get profile by index."""
        if 0 <= index < len(self.profiles):
            return self.profiles[index]
        return None
    
    def set_num_types(self, n_types: int):
        """Resize registry for new type count."""
        current = len(self.profiles)
        
        if n_types > current:
            # Add new profiles with random parameters
            for i in range(current, n_types):
                self.add_profile(WaveProfile(
                    name=f"Species_{i}",
                    frequency=float(np.random.randint(2, 8)),
                    amplitude=np.random.uniform(0.01, 0.04),
                    phase_speed=np.random.uniform(-3.0, 3.0),
                ))
        elif n_types < current:
            self.profiles = self.profiles[:n_types]
        
        self._interaction_matrix = None
    
    def _rebuild_interaction_matrix(self):
        """Rebuild the type-to-type wave interaction matrix."""
        n = len(self.profiles)
        if n == 0:
            self._interaction_matrix = np.zeros((1, 1), dtype=np.float32)
            return
        
        mat = np.zeros((n, n), dtype=np.float32)
        
        for i, pi in enumerate(self.profiles):
            for j, pj in enumerate(self.profiles):
                if i == j:
                    mat[i, j] = pi.self_interference
                else:
                    # Average of cross-interference values
                    mat[i, j] = 0.5 * (pi.cross_interference + pj.cross_interference)
        
        self._interaction_matrix = mat
    
    def get_interaction_matrix(self) -> np.ndarray:
        """Get the wave interaction strength matrix."""
        if self._interaction_matrix is None:
            self._rebuild_interaction_matrix()
        return self._interaction_matrix
    
    def pack_for_kernel(self) -> tuple:
        """
        Pack wave parameters for Numba kernels.
        
        Returns:
            (frequencies, amplitudes, phase_speeds, interaction_matrix)
        """
        n = len(self.profiles)
        if n == 0:
            return (
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )
        
        freqs = np.array([p.frequency for p in self.profiles], dtype=np.float32)
        amps = np.array([p.amplitude for p in self.profiles], dtype=np.float32)
        speeds = np.array([p.phase_speed for p in self.profiles], dtype=np.float32)
        
        return freqs, amps, speeds, self.get_interaction_matrix()
    
    def to_dict(self) -> dict:
        """Serialize registry to dictionary."""
        return {
            'config': {
                'mode': self.config.mode,
                'repulsion_strength': self.config.repulsion_strength,
                'repulsion_exponent': self.config.repulsion_exponent,
                'constructive_multiplier': self.config.constructive_multiplier,
                'destructive_multiplier': self.config.destructive_multiplier,
            },
            'profiles': [
                {
                    'name': p.name,
                    'frequency': p.frequency,
                    'amplitude': p.amplitude,
                    'phase_speed': p.phase_speed,
                    'self_interference': p.self_interference,
                    'cross_interference': p.cross_interference,
                }
                for p in self.profiles
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WaveRegistry':
        """Deserialize from dictionary."""
        config_data = data.get('config', {})
        config = WaveConfig(
            mode=WaveMode(config_data.get('mode', 1)),
            repulsion_strength=config_data.get('repulsion_strength', 30.0),
            repulsion_exponent=config_data.get('repulsion_exponent', 8.0),
            constructive_multiplier=config_data.get('constructive_multiplier', 1.5),
            destructive_multiplier=config_data.get('destructive_multiplier', 0.5),
        )
        
        profiles = [
            WaveProfile(
                name=p['name'],
                frequency=p.get('frequency', 3.0),
                amplitude=p.get('amplitude', 0.02),
                phase_speed=p.get('phase_speed', 1.0),
                self_interference=p.get('self_interference', 1.0),
                cross_interference=p.get('cross_interference', 0.5),
            )
            for p in data.get('profiles', [])
        ]
        
        return cls(profiles=profiles, config=config)


def create_wave_preset(preset: str, n_types: int = 6) -> WaveRegistry:
    """
    Create a wave registry with preset configurations.
    
    Presets:
    - 'calm': Low amplitude, slow phase
    - 'active': High amplitude, fast phase
    - 'interference': Optimized for interference patterns
    - 'standing': Optimized for standing wave formation
    - 'asymmetric': Mix of left and right handed waves
    
    Args:
        preset: Preset name
        n_types: Number of species
        
    Returns:
        Configured WaveRegistry
    """
    registry = WaveRegistry()
    
    if preset == 'calm':
        config = WaveConfig(
            mode=WaveMode.STANDARD,
            repulsion_strength=15.0,
            repulsion_exponent=5.0,
        )
        for i in range(n_types):
            registry.add_profile(WaveProfile(
                name=f"Calm_{i}",
                frequency=float(np.random.randint(2, 4)),
                amplitude=np.random.uniform(0.005, 0.015),
                phase_speed=np.random.uniform(-0.5, 0.5),
            ))
    
    elif preset == 'active':
        config = WaveConfig(
            mode=WaveMode.INTERFERENCE,
            repulsion_strength=50.0,
            repulsion_exponent=12.0,
        )
        for i in range(n_types):
            registry.add_profile(WaveProfile(
                name=f"Active_{i}",
                frequency=float(np.random.randint(4, 8)),
                amplitude=np.random.uniform(0.03, 0.06),
                phase_speed=np.random.uniform(-4.0, 4.0),
            ))
    
    elif preset == 'interference':
        config = WaveConfig(
            mode=WaveMode.INTERFERENCE,
            repulsion_strength=30.0,
            repulsion_exponent=8.0,
            constructive_multiplier=2.0,
            destructive_multiplier=0.3,
        )
        # Create pairs with matching frequencies for interference
        for i in range(n_types):
            freq = float(3 + (i % 3))  # Groups of similar frequency
            registry.add_profile(WaveProfile(
                name=f"Interfere_{i}",
                frequency=freq,
                amplitude=0.025,
                phase_speed=np.random.uniform(-2.0, 2.0),
                self_interference=1.5,
                cross_interference=0.8,
            ))
    
    elif preset == 'standing':
        config = WaveConfig(
            mode=WaveMode.STANDING,
            repulsion_strength=25.0,
            repulsion_exponent=6.0,
            standing_wave_threshold=0.7,
            standing_wave_damping=0.98,
        )
        # Uniform frequencies for standing wave formation
        for i in range(n_types):
            registry.add_profile(WaveProfile(
                name=f"Standing_{i}",
                frequency=4.0,  # All same frequency
                amplitude=0.02,
                phase_speed=np.random.choice([-1.0, 1.0]),  # Opposite directions
            ))
    
    elif preset == 'asymmetric':
        config = WaveConfig(
            mode=WaveMode.INTERFERENCE,
            repulsion_strength=35.0,
            repulsion_exponent=10.0,
            quadrature_torque=0.5,  # Strong torque from phase differences
        )
        for i in range(n_types):
            registry.add_profile(WaveProfile(
                name=f"Asym_{i}",
                frequency=float(np.random.randint(3, 6)),
                amplitude=np.random.uniform(0.02, 0.04),
                phase_speed=np.random.uniform(-3.0, 3.0),
                chirality=1 if i % 2 == 0 else -1,  # Alternating handedness
            ))
    
    else:
        # Default
        config = WaveConfig()
        for i in range(n_types):
            registry.add_profile(WaveProfile(
                name=f"Default_{i}",
                frequency=float(np.random.randint(2, 6)),
                amplitude=np.random.uniform(0.01, 0.03),
                phase_speed=np.random.uniform(-2.0, 2.0),
            ))
    
    registry.config = config
    return registry
