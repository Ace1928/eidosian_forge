"""
Eidosian PyParticles V6 - Exclusion Registry

Manages exclusion presets and configurations.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .types import ExclusionConfig, SpinConfig, ParticleBehavior


@dataclass
class ExclusionPreset:
    """Named preset for exclusion mechanics."""
    name: str
    description: str
    exclusion_config: ExclusionConfig
    spin_config: Optional[SpinConfig] = None


class ExclusionRegistry:
    """
    Registry for exclusion mechanics configurations.
    
    Provides presets for common quantum-inspired behaviors:
    - fermi_gas: All particles fermionic with exclusion
    - bose_einstein: All particles bosonic, can condense
    - electron_gas: Spin-1/2 fermions with Cooper pairing
    - mixed: Some fermionic, some bosonic
    """
    
    def __init__(self, n_types: int):
        self.n_types = n_types
        self.presets: Dict[str, ExclusionPreset] = {}
        self._init_default_presets()
        
        # Active configuration
        self.config = ExclusionConfig()
        self.spin_config = SpinConfig.default(n_types)
        self.config.initialize(n_types)
    
    def _init_default_presets(self):
        """Initialize built-in presets."""
        n = self.n_types
        
        # 1. Fermi Gas - strong exclusion for all
        fermi_cfg = ExclusionConfig.all_fermionic(n, strength=25.0)
        fermi_spin = SpinConfig.default(n)
        self.presets['fermi_gas'] = ExclusionPreset(
            name='Fermi Gas',
            description='All particles are fermions with Pauli exclusion',
            exclusion_config=fermi_cfg,
            spin_config=fermi_spin,
        )
        
        # 2. Bose-Einstein - no exclusion, can condense
        bose_cfg = ExclusionConfig.all_bosonic(n)
        bose_spin = SpinConfig.default(n)
        bose_spin.spin_enabled[:] = False  # Bosons are spinless
        self.presets['bose_einstein'] = ExclusionPreset(
            name='Bose-Einstein',
            description='All particles are bosons, can overlap and condense',
            exclusion_config=bose_cfg,
            spin_config=bose_spin,
        )
        
        # 3. Electron Gas - spin-1/2 with pairing
        electron_cfg = ExclusionConfig.all_fermionic(n, strength=30.0)
        electron_spin = SpinConfig.default(n)
        electron_spin.flip_threshold[:] = 1.5  # Easier spin flips
        electron_spin.coupling_strength[:] = 1.5  # Strong coupling
        self.presets['electron_gas'] = ExclusionPreset(
            name='Electron Gas',
            description='Spin-1/2 fermions with Cooper pairing',
            exclusion_config=electron_cfg,
            spin_config=electron_spin,
        )
        
        # 4. Mixed - realistic mixture
        mixed_cfg = ExclusionConfig()
        mixed_cfg.exclusion_strength = 20.0
        mixed_cfg.initialize(n)
        mixed_spin = SpinConfig.mixed(n)
        self.presets['mixed'] = ExclusionPreset(
            name='Mixed',
            description='Mix of fermionic and bosonic particles',
            exclusion_config=mixed_cfg,
            spin_config=mixed_spin,
        )
        
        # 5. Classical - no quantum effects
        classical_cfg = ExclusionConfig()
        classical_cfg.exclusion_strength = 0.0
        classical_cfg.type_behavior = np.full(n, ParticleBehavior.CLASSICAL, dtype=np.int32)
        classical_cfg.behavior_matrix = np.full((n, n), ParticleBehavior.CLASSICAL, dtype=np.int32)
        classical_spin = SpinConfig.default(n)
        classical_spin.spin_enabled[:] = False
        self.presets['classical'] = ExclusionPreset(
            name='Classical',
            description='No quantum exclusion effects',
            exclusion_config=classical_cfg,
            spin_config=classical_spin,
        )
    
    def apply_preset(self, preset_name: str):
        """Apply a named preset to the active configuration."""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available: {list(self.presets.keys())}")
        
        preset = self.presets[preset_name]
        self.config = preset.exclusion_config
        if preset.spin_config is not None:
            self.spin_config = preset.spin_config
    
    def resize(self, n_types: int):
        """Resize for new type count."""
        if n_types == self.n_types:
            return
        
        self.n_types = n_types
        self._init_default_presets()
        self.config.initialize(n_types)
        self.spin_config = SpinConfig.default(n_types)
    
    def pack_for_kernel(self) -> dict:
        """
        Pack configuration into arrays for Numba kernels.
        
        Returns dict with:
            'behavior_matrix': (T, T) int32
            'exclusion_strength': float
            'exclusion_radius_factor': float
            'spin_enabled': (T,) bool
            'flip_threshold': (T,) float32
            'flip_probability': (T,) float32
            'coupling_strength': (T,) float32
        """
        return {
            'behavior_matrix': self.config.behavior_matrix,
            'exclusion_strength': self.config.exclusion_strength,
            'exclusion_radius_factor': self.config.exclusion_radius_factor,
            'spin_enabled': self.spin_config.spin_enabled,
            'flip_threshold': self.spin_config.flip_threshold,
            'flip_probability': self.spin_config.flip_probability,
            'coupling_strength': self.spin_config.coupling_strength,
        }
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'n_types': self.n_types,
            'exclusion_strength': self.config.exclusion_strength,
            'exclusion_radius_factor': self.config.exclusion_radius_factor,
            'allow_spin_flips': self.config.allow_spin_flips,
            'type_behavior': self.config.type_behavior.tolist(),
            'spin_enabled': self.spin_config.spin_enabled.tolist(),
            'flip_threshold': self.spin_config.flip_threshold.tolist(),
            'coupling_strength': self.spin_config.coupling_strength.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExclusionRegistry':
        """Deserialize from dictionary."""
        n_types = data['n_types']
        reg = cls(n_types)
        
        reg.config.exclusion_strength = data.get('exclusion_strength', 20.0)
        reg.config.exclusion_radius_factor = data.get('exclusion_radius_factor', 2.0)
        reg.config.allow_spin_flips = data.get('allow_spin_flips', True)
        
        if 'type_behavior' in data:
            reg.config.type_behavior = np.array(data['type_behavior'], dtype=np.int32)
            reg.config.initialize(n_types)
        
        if 'spin_enabled' in data:
            reg.spin_config.spin_enabled = np.array(data['spin_enabled'], dtype=np.bool_)
        if 'flip_threshold' in data:
            reg.spin_config.flip_threshold = np.array(data['flip_threshold'], dtype=np.float32)
        if 'coupling_strength' in data:
            reg.spin_config.coupling_strength = np.array(data['coupling_strength'], dtype=np.float32)
        
        return reg
