# Stratum - Emergent Layered Physical Simulation Engine

Stratum is a production-grade simulation engine designed to model emergent, layered physical phenomena. It provides a deterministic, conservation-aware framework for simulating complex physical systems with multiple interacting scales.

## Key Features

- **Deterministic Execution**: Reproducible results with blake2s-based entropy, configurable determinism modes
- **Conservation Accounting**: Track mass, energy, and momentum with source/sink ledger
- **Event System**: Subscribe to simulation events for game integration
- **Boundary-Aware**: PERIODIC, REFLECTIVE, and OPEN boundary conditions with universal enforcement
- **High Performance**: Cached derived fields, efficient top-K selection, patch-based scheduling
- **Extensible Architecture**: Plugin interfaces for domain packs and custom physics

## Quick Start

```bash
# Install from source
pip install -e .

# Run the unified scenario launcher (from game_forge/src on PYTHONPATH)
PYTHONPATH=src python -m Stratum --scenario collapse --grid 32 --ticks 100

# Run the stellar collapse demo
python -m scenarios.stellar_collapse --grid 32 --ticks 100

# Run the real-time screensaver
python -m scenarios.stellar_screensaver --grid 128

# Verify determinism
python tests/golden_checksum.py --seed 42 --ticks 100 --runs 3
```

## Installation

### Requirements

- Python 3.9+
- NumPy >= 1.20.0

### Optional Dependencies

- Matplotlib (for snapshot visualization)
- Pygame (for real-time screensaver)

### Install

```bash
# Basic installation
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With development tools
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## Architecture

```
stratum/
├── core/                  # Core engine components
│   ├── config.py          # EngineConfig with determinism modes
│   ├── conservation.py    # Conservation ledger (mass/energy/momentum)
│   ├── events.py          # EventBus for game integration
│   ├── fabric.py          # Field storage with boundary handling
│   ├── ledger.py          # Energy accounting with blake2s entropy
│   ├── metronome.py       # Timing and budget allocation
│   ├── ordering.py        # Canonical iteration ordering
│   ├── quanta.py          # Microtick resolution with cached fields
│   ├── registry.py        # Species registry with versioning
│   └── types.py           # Core type definitions
├── domains/               # Domain-specific physics
│   └── materials/         # High-energy materials subsystem
├── scenarios/             # Runnable scenarios
│   ├── stellar_collapse.py
│   ├── stellar_collapse_runtime.py
│   └── stellar_screensaver.py
└── tests/                 # Test suite (212+ tests)
    ├── test_determinism.py
    ├── test_properties.py
    └── golden_checksum.py
```

## Determinism Modes

Stratum provides three determinism levels:

```python
from core.config import EngineConfig, DeterminismMode

# Bitwise-identical results (same seed + version = same checksum)
config = EngineConfig(
    determinism_mode=DeterminismMode.STRICT_DETERMINISTIC,
    entropy_mode=False,
)

# Statistically reproducible (allows adaptive scheduling)
config = EngineConfig(
    determinism_mode=DeterminismMode.REPLAY_DETERMINISTIC,
)

# Maximum performance (may vary between runs)
config = EngineConfig(
    determinism_mode=DeterminismMode.REALTIME_ADAPTIVE,
    entropy_mode=True,
)
```

### Verify Determinism

```bash
# Run golden checksum test
python tests/golden_checksum.py --seed 42 --ticks 100 --runs 3

# Output:
# ✅ PASS: All runs produced identical checksums
#    Final state: da6fc0c104454a9f94b5bac57f986d3804f32f7f1c32b556597516e5d578ffdf
```

## Conservation Tracking

Track mass, energy, and momentum conservation:

```python
from core.conservation import ConservationLedger

ledger = ConservationLedger(tolerance=1e-6)

# In simulation loop
ledger.begin_tick(tick, fabric)
# ... simulation step ...
ledger.record_boundary_flux(tick, cell, mass_delta, energy_delta)
ledger.record_bh_absorption(tick, cell, mass, energy)
stats = ledger.end_tick(tick, fabric)

# Check conservation
is_ok, msg = ledger.check_conservation()
print(f"Conservation: {msg}")
```

## Event System

Subscribe to simulation events for game integration:

```python
from core.events import EventBus, EventType

bus = EventBus(buffer_size=10000)

def on_bh_formed(event):
    print(f"Black hole formed at {event.cell} with mass {event.data['initial_mass']}")

bus.subscribe(EventType.BLACK_HOLE_FORMED, on_bh_formed)

# In simulation, emit events
bus.emit(EventType.BLACK_HOLE_FORMED, tick=100, cell=(5, 5), 
         data={'initial_mass': 10.0, 'trigger_Z': 4.5})

# Query events
recent_bh = bus.get_events(EventType.BLACK_HOLE_FORMED, limit=10)
```

## Configuration

```python
from core.config import EngineConfig, DeterminismMode, NegativeDensityPolicy

config = EngineConfig(
    # Grid
    grid_w=128,
    grid_h=128,
    dx=1.0,  # Grid spacing
    dt_tick=1.0,  # Time per tick
    
    # Determinism
    determinism_mode=DeterminismMode.STRICT_DETERMINISTIC,
    base_seed=42,
    
    # Boundaries
    boundary="PERIODIC",  # PERIODIC, REFLECTIVE, or OPEN
    negative_density_policy=NegativeDensityPolicy.CLAMP_ZERO,
    
    # Stability
    cfl_safety_factor=0.5,
    diffusion_stability_limit=0.25,
    
    # Physics
    gravity_strength=0.05,
    thermal_pressure_coeff=0.1,
    
    # Conservation
    enforce_mass_conservation=False,
    enforce_energy_bounds=True,
    
    # World law version (for replay compatibility)
    world_law_version="1.0.0",
)
```

## Scenarios

### Stellar Collapse

```bash
python -m scenarios.stellar_collapse \
    --grid 64 \
    --ticks 500 \
    --microticks 200 \
    --output ./outputs
```

### Runtime-Controlled

```bash
python -m scenarios.stellar_collapse_runtime \
    --grid 64 \
    --runtime 60 \
    --snapshot 2.0
```

### Interactive Screensaver

```bash
python -m scenarios.stellar_screensaver \
    --grid 256 \
    --fps 60 \
    --lod 1.0

# Controls:
#   ESC/Q: Quit
#   SPACE: Pause
#   R: Restart
#   TAB: Toggle log scaling
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=domains --cov-report=term

# Run specific test categories
pytest tests/test_determinism.py -v  # Determinism verification
pytest tests/test_properties.py -v   # Property-based tests
pytest tests/test_integration.py -v  # Integration tests
```

## CI/CD

GitHub Actions workflow runs:
- Tests on Python 3.9, 3.10, 3.11, 3.12
- Linting with ruff
- Type checking with mypy

## Development

### Adding New Physics

1. Create module in `domains/`
2. Implement operator interface
3. Hook into Quanta phase pipeline
4. Add tests

### Adding New Events

1. Add event type to `EventType` enum
2. Create helper function in `events.py`
3. Emit from appropriate location

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests (212+ and counting!)
4. Ensure determinism tests pass
5. Submit a pull request
