# Stratum TODO List

This document outlines planned improvements, features, and technical debt items for the Stratum simulation engine.

## High Priority

### Performance Optimizations

- [ ] **Vectorize global operations**: Replace Python loops in `apply_global_ops()` with NumPy vectorized operations for heat/radiation diffusion
- [ ] **Optimize effective property calculation**: Cache mixture-weighted properties per tick instead of recalculating per cell
- [ ] **Implement spatial partitioning**: Use quadtree or similar structure for efficient neighbor queries
- [ ] **Add Numba JIT compilation**: Accelerate tight loops in `resolve_cell()` and mass transfer

### Core Engine Improvements

- [ ] **Implement proper signal propagation**: Current signals only affect origin cell; extend to radius-based footprints
- [ ] **Add distance-based signal attenuation**: Implement proper 1/r or 1/r² falloff for signals
- [ ] **Improve mass conservation**: Address numerical drift in mass transfer operations
- [ ] **Add energy conservation tracking**: Implement global energy audit per tick

### Physics Fidelity

- [ ] **Improve EOS model**: Replace simple polytropic EOS with more realistic equation of state
- [ ] **Add nuclear binding energy curve**: Implement semi-empirical mass formula for fusion energetics
- [ ] **Implement proper shock capturing**: Add artificial viscosity for discontinuity handling
- [ ] **Add magnetic field support**: Extend Fabric with B-field arrays for MHD effects

## Medium Priority

### New Features

- [ ] **Chemistry domain**: Implement low-energy chemical reactions
  - Bond formation/breaking
  - Catalysis effects
  - Temperature-dependent reaction rates
- [ ] **Rendering subsystem**: Create dedicated render package
  - Multiple visualization modes (density, temperature, velocity)
  - Vector field visualization
  - Species composition coloring
- [ ] **Event logging**: Add comprehensive event log for analysis
  - Fusion events
  - Decay events
  - Black hole formations
  - Energy exchanges
- [ ] **Checkpoint/restore**: Implement simulation state serialization
  - Save full state to disk
  - Resume from checkpoint
  - State comparison tools

### Developer Experience

- [ ] **Add type stubs**: Create .pyi files for better IDE support
- [ ] **Improve error messages**: Add descriptive error messages with suggested fixes
- [ ] **Add logging framework**: Replace print statements with structured logging
- [ ] **Create debugging tools**: Add visualization of intermediate states during microticks

### Documentation

- [ ] **Add API documentation**: Generate comprehensive API docs (Sphinx or similar)
- [ ] **Create tutorials**: Step-by-step guides for common use cases
- [ ] **Add architecture diagrams**: Visual documentation of system design
- [ ] **Document physics models**: Mathematical descriptions of implemented equations

## Low Priority

### Future Enhancements

- [ ] **GPU acceleration**: Implement CUDA/OpenCL kernels for parallel computation
- [ ] **Distributed simulation**: Support for multi-node simulation clusters
- [ ] **Web interface**: Browser-based visualization and control
- [ ] **VR/AR visualization**: 3D immersive visualization support

### Code Quality

- [ ] **Add static type checking**: Run mypy on codebase and fix issues
- [ ] **Increase test coverage**: Target 95%+ line coverage
- [ ] **Add property-based tests**: Use hypothesis for fuzzing
- [ ] **Benchmark suite**: Automated performance regression testing

### Configuration

- [ ] **YAML/TOML config files**: Support external configuration files
- [ ] **Config validation**: Add schema validation for configurations
- [ ] **Config presets**: Pre-built configurations for common scenarios
- [ ] **Runtime config modification**: Hot-reload of parameters during simulation

## Technical Debt

### Known Issues

- [ ] **Overflow warnings**: Address RuntimeWarning in kinetic energy calculations for extreme values
- [ ] **Boundary handling edge cases**: Review and test all boundary conditions thoroughly
- [ ] **Registry concurrency**: Add file locking for multi-process registry access
- [ ] **Memory usage**: Profile and optimize memory allocation patterns

### Refactoring

- [ ] **Decouple Materials from Quanta**: Reduce coupling between subsystems
- [ ] **Extract barrier calculations**: Create dedicated barrier module
- [ ] **Standardize property naming**: Consistent naming convention across all properties
- [ ] **Modularize scenarios**: Create scenario base class and plugin system

### Testing

- [ ] **Add stress tests**: Long-running simulations to find edge cases
- [ ] **Add fuzzing tests**: Random input generation for robustness
- [ ] **Add performance tests**: Ensure no performance regressions
- [ ] **Add visual regression tests**: Compare output images across versions

## Completed

- [x] Add math import to quanta.py and fundamentals.py
- [x] Create comprehensive test suite (197 tests)
- [x] Create README documentation
- [x] Create TODO documentation
- [x] Add conftest.py for test configuration
- [x] Fix relative imports for testing compatibility
