"""
Core engine package for the Stratum simulation.

This package contains the fundamental components of the Stratum simulation
engine:

- ``config``: Simulation configuration with EngineConfig dataclass
- ``fabric``: Spatial field storage and boundary handling
- ``ledger``: Energy conservation, barrier calculations, and entropy source
- ``metronome``: Timing and compute budget allocation
- ``quanta``: Event propagation and microtick resolution
- ``registry``: Species registry with property quantization
- ``types``: Core type definitions (Vec2, Cell, utility functions)

Example usage::

    from core.config import EngineConfig
    from core.fabric import Fabric
    from core.ledger import Ledger
    from core.quanta import Quanta
    from core.registry import SpeciesRegistry

    # Create configuration
    cfg = EngineConfig(grid_w=64, grid_h=64)
    
    # Initialize components
    fabric = Fabric(cfg)
    ledger = Ledger(fabric, cfg)
    # ... etc.
"""

from .config import EngineConfig
from .types import Vec2, Cell, dot, clamp
from .fabric import Fabric, Mixture
from .ledger import Ledger, EntropySource
from .metronome import Metronome, MetronomeStats
from .registry import SpeciesRegistry, Species

__all__ = [
    "EngineConfig",
    "Vec2",
    "Cell",
    "dot",
    "clamp",
    "Fabric",
    "Mixture",
    "Ledger",
    "EntropySource",
    "Metronome",
    "MetronomeStats",
    "SpeciesRegistry",
    "Species",
]
