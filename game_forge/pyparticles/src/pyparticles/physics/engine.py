"""
Eidosian PyParticles V6 - Physics Engine

The core physics engine orchestrating:
- Multiple interaction rules with pluggable force types
- Velocity Verlet symplectic integration
- Spatial hashing for O(N) neighbor lookup
- Exclusion mechanics (Pauli-like repulsion)
- Spin dynamics with stochastic flips
- Berendsen thermostat for temperature control

Architecture:
    PhysicsEngine
    ├── ParticleState (positions, velocities, angles, colors)
    ├── InteractionRule[] (force matrices + parameters)
    ├── SpeciesConfig (radii, wave properties, spin dynamics)
    ├── ExclusionMechanics (spin states, behavior matrix)
    └── SpatialGrid (grid_counts, grid_cells)

Usage:
    >>> cfg = SimulationConfig(num_particles=5000, num_types=8)
    >>> engine = PhysicsEngine(cfg)
    >>> for _ in range(100):
    ...     engine.update()
    >>> stats = engine.get_spin_statistics()
    
Performance:
    - 1K particles: ~1000 FPS
    - 5K particles: ~70 FPS  
    - 10K particles: ~19 FPS

Author: Eidosian Framework
"""
import numpy as np
from typing import List, Optional
from ..core.types import (
    ParticleState, SimulationConfig, InteractionRule, ForceType, 
    SpeciesConfig, SpinInteractionMatrix
)
from .kernels import (
    fill_grid, compute_forces_multi, apply_thermostat, 
    integrate_verlet_1, integrate_verlet_2
)
from .exclusion.kernels import (
    apply_exclusion_forces_wave, apply_spin_flip, apply_spin_coupling,
    compute_spin_statistics, initialize_spins
)
from .exclusion.types import ParticleBehavior


class PhysicsEngine:
    """
    High-performance particle physics engine with advanced mechanics.
    
    This engine implements a complete particle simulation with:
    - Multiple configurable force types (Linear, Yukawa, LJ, etc.)
    - Quantum-inspired exclusion mechanics
    - Spin state dynamics
    - Wave mechanics integration
    - Symplectic (energy-conserving) integration
    
    Attributes:
        cfg: SimulationConfig with all parameters
        state: ParticleState container (pos, vel, angle, etc.)
        spin: Per-particle spin states (-1, 0, +1)
        rules: List of InteractionRule definitions
        species_config: Per-species parameters (radius, wave, spin)
        behavior_matrix: Fermionic/Bosonic behavior per type pair
        
    Example:
        >>> engine = PhysicsEngine(SimulationConfig(num_particles=1000))
        >>> engine.exclusion_enabled = True
        >>> engine.exclusion_strength = 10.0
        >>> for _ in range(60):  # 1 second at 60 FPS
        ...     engine.update(dt=1/60)
    """
    
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        
        # State Container - with spin array
        self.state = ParticleState.allocate(config.max_particles)
        self.state.active = config.num_particles
        
        # Spin states for each particle (-1=down, 0=none, +1=up)
        self.spin = np.zeros(config.max_particles, dtype=np.int8)
        
        # Exclusion mechanics settings
        # Lower default strength - was causing chaos when enabled
        self.exclusion_enabled = True
        self.exclusion_strength = 2.0  # Reduced from 8.0 - much gentler
        self.exclusion_radius_factor = 2.0  # Reduced from 3.0
        
        # Spin flip settings
        self.spin_flip_enabled = True
        self.spin_flip_threshold = 5.0
        self.spin_flip_probability = 0.1
        
        # Spin coupling settings (for angular velocity alignment)
        self.spin_enabled = True
        self.spin_coupling_strength = 0.5
        self.spin_interaction_range = config.world_size * 0.05  # 5% of world
        
        # Frame counter for spin flip RNG
        self._frame = 0
        
        # Initialize Rules
        self.rules: List[InteractionRule] = []
        self._init_default_rules()
        
        # Species Config - scaled to world size
        self.species_config = SpeciesConfig.default(config.num_types, config.world_size)
        
        # Spin interaction matrix
        self.spin_matrix = SpinInteractionMatrix.default(config.num_types)
        
        # Behavior matrix - default to fermionic for all type pairs
        self.behavior_matrix = np.ones(
            (config.num_types, config.num_types), dtype=np.int32
        ) * ParticleBehavior.FERMIONIC.value
        
        # Spin coupling matrix (strength between type pairs for angular dynamics)
        self.spin_coupling_matrix = np.ones(
            (config.num_types, config.num_types), dtype=np.float32
        )
        
        # Initialize Random Particles
        self._init_particles()
        
        # Grid Memory - scaled to world size
        self.max_interaction_radius = self._get_max_radius()
        self.cell_size = max(self.max_interaction_radius, 0.1 * config.world_size)
        
        self.grid_w = int(config.world_size / self.cell_size) + 2
        self.grid_h = int(config.world_size / self.cell_size) + 2
        
        avg_density = config.num_particles / (self.grid_w * self.grid_h)
        self.max_per_cell = int(avg_density * 20) + 100 
        
        self.grid_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.grid_cells = np.zeros((self.grid_h, self.grid_w, self.max_per_cell), dtype=np.int32)
        
    def _init_default_rules(self):
        """Initialize default interaction rules scaled to world size."""
        max_r = self.cfg.default_max_radius
        min_r = self.cfg.default_min_radius
        
        # Main particle life force (linear dropoff) - weaker for stability
        mat_linear = np.random.uniform(-0.5, 0.5, 
            (self.cfg.num_types, self.cfg.num_types)).astype(np.float32)
        # Make diagonal slightly attractive (same-type cohesion)
        for i in range(self.cfg.num_types):
            mat_linear[i, i] = np.random.uniform(0.1, 0.3)
        
        rule_lin = InteractionRule(
            name="Particle Life (Linear)",
            force_type=ForceType.LINEAR,
            matrix=mat_linear,
            max_radius=max_r,
            min_radius=min_r,
            strength=0.8  # Reduced for stability
        )
        self.rules.append(rule_lin)
        
        # Long-range gravity-like attraction - very weak
        mat_grav = np.full((self.cfg.num_types, self.cfg.num_types), 
                          0.01, dtype=np.float32)  # Very weak universal attraction
        rule_grav = InteractionRule(
            name="Gravity (InvSq)",
            force_type=ForceType.INVERSE_SQUARE,
            matrix=mat_grav,
            max_radius=max_r * 3,  # Longer range
            min_radius=min_r,
            strength=0.1,  # Very weak
            softening=min_r * 3
        )
        self.rules.append(rule_grav)
        
        # STRONG exclusion repulsion (prevents overlap) - CRITICAL
        mat_repel = np.full((self.cfg.num_types, self.cfg.num_types),
                           1.0, dtype=np.float32)
        rule_repel = InteractionRule(
            name="Exclusion Repulsion",
            force_type=ForceType.REPEL_ONLY,
            matrix=mat_repel,
            max_radius=min_r * 8,  # Wider exclusion zone
            min_radius=min_r * 0.1,
            strength=15.0,  # VERY STRONG repulsion
            softening=min_r * 0.5
        )
        self.rules.append(rule_repel)
    
    def setup_classic_rules(self):
        """
        Setup EXACT Haskell particle-life attraction matrix.
        
        This replaces the random rules with the specific matrix from
        the original Haskell implementation for authentic emergence.
        
        Haskell attraction matrix (4 species: Red, Green, Blue, Yellow):
          Red-Red: 0.6, Red-Green: -0.3, Red-Blue: 0.2, Red-Yellow: -0.4
          Green-Green: 2.0, Green-Blue: 0.3, Green-Yellow: -0.2
          Blue-Blue: -0.4, Blue-Yellow: 0.5
          Yellow-Yellow: -0.2
        """
        if self.cfg.num_types != 4:
            print("[Warning] Classic rules expect 4 types, got {}".format(self.cfg.num_types))
        
        # Clear existing rules
        self.rules = []
        
        max_r = self.cfg.default_max_radius
        min_r = self.cfg.default_min_radius
        
        # Build the exact Haskell attraction matrix
        mat = np.zeros((4, 4), dtype=np.float32)
        # Red=0, Green=1, Blue=2, Yellow=3
        mat[0, 0] =  0.6   # Red-Red
        mat[0, 1] = -0.3   # Red-Green
        mat[0, 2] =  0.2   # Red-Blue
        mat[0, 3] = -0.4   # Red-Yellow
        mat[1, 1] =  2.0   # Green-Green
        mat[1, 2] =  0.3   # Green-Blue
        mat[1, 3] = -0.2   # Green-Yellow
        mat[2, 2] = -0.4   # Blue-Blue
        mat[2, 3] =  0.5   # Blue-Yellow
        mat[3, 3] = -0.2   # Yellow-Yellow
        # Symmetric
        mat[1, 0] = mat[0, 1]
        mat[2, 0] = mat[0, 2]
        mat[2, 1] = mat[1, 2]
        mat[3, 0] = mat[0, 3]
        mat[3, 1] = mat[1, 3]
        mat[3, 2] = mat[2, 3]
        
        # Single LINEAR force rule (exactly like Haskell)
        rule_lin = InteractionRule(
            name="Classic Particle Life",
            force_type=ForceType.LINEAR,
            matrix=mat,
            max_radius=max_r,
            min_radius=min_r,
            strength=1.0  # Use matrix values directly
        )
        self.rules.append(rule_lin)
        
        # CRITICAL: Disable wave visuals for classic mode (Haskell doesn't have waves)
        n = self.cfg.num_types
        self.species_config.wave_amp = np.zeros(n, dtype=np.float32)
        self.species_config.wave_freq = np.ones(n, dtype=np.float32)  # Circular shape
        
        # Rebuild grid with new max radius
        self.max_interaction_radius = self._get_max_radius()
        self.cell_size = max(self.max_interaction_radius, 0.1 * self.cfg.world_size)
        self._invalidate_cache()
        print("[Classic] Haskell attraction matrix loaded: 4 species, max_r={:.2f}, min_r={:.2f}".format(max_r, min_r))

    def _get_max_radius(self):
        if not self.rules: 
            return self.cfg.default_max_radius
        return max(r.max_radius for r in self.rules)

    def _init_particles(self):
        """Initialize particles with species-driven properties and spin."""
        n = self.state.active
        half = self.cfg.half_world
        
        # Random positions in world
        self.state.pos[:n] = np.random.uniform(-half, half, (n, 2)).astype(np.float32)
        self.state.vel[:n] = 0.0
        self.state.colors[:n] = np.random.randint(0, self.cfg.num_types, n)
        self.state.angle[:n] = np.random.uniform(0, 2*np.pi, n).astype(np.float32)
        
        # Initialize angular velocity from species base_spin_rate
        for i in range(n):
            t = self.state.colors[i]
            self.state.ang_vel[i] = self.species_config.base_spin_rate[t]
        
        # Initialize spins (random up/down for all particles)
        spin_enabled = np.ones(self.cfg.num_types, dtype=np.bool_)
        initialize_spins(self.spin, self.state.colors, spin_enabled, n, seed=42)

    def reset(self):
        self._init_particles()
        self._invalidate_cache()

    def set_active_count(self, count: int):
        if count > self.cfg.max_particles: count = self.cfg.max_particles
        old_count = self.state.active
        self.state.active = count
        half = self.cfg.half_world
        
        if count > old_count:
            diff = count - old_count
            self.state.pos[old_count:count] = np.random.uniform(-half, half, (diff, 2)).astype(np.float32)
            self.state.vel[old_count:count] = 0.0
            self.state.colors[old_count:count] = np.random.randint(0, self.cfg.num_types, diff)
            self.state.angle[old_count:count] = np.random.uniform(0, 2*np.pi, diff).astype(np.float32)
            for i in range(old_count, count):
                t = self.state.colors[i]
                self.state.ang_vel[i] = self.species_config.base_spin_rate[t]
            # Initialize spins for new particles
            self.spin[old_count:count] = np.random.choice(
                np.array([-1, 1], dtype=np.int8), diff
            )
        self._invalidate_cache()

    def set_species_count(self, n_types: int):
        if n_types == self.cfg.num_types: return
        self.cfg.num_types = n_types
        self.species_config = SpeciesConfig.default(n_types, self.cfg.world_size)
        self.spin_matrix = SpinInteractionMatrix.default(n_types)
        self.behavior_matrix = np.ones(
            (n_types, n_types), dtype=np.int32
        ) * ParticleBehavior.FERMIONIC.value
        self.rules = []
        self._init_default_rules()
        self._invalidate_cache()
        self.reset()
    
    def _invalidate_cache(self):
        """Invalidate cached forces/torques to force recomputation."""
        if hasattr(self, 'forces_cache'):
            delattr(self, 'forces_cache')
        if hasattr(self, 'torques_cache'):
            delattr(self, 'torques_cache')

    def _pack_rules(self):
        """Pack rules into arrays for Numba kernel (8-column format)."""
        active_rules = [r for r in self.rules if r.enabled]
        n_rules = len(active_rules)
        n_types = self.cfg.num_types
        
        if n_rules == 0:
            # Return minimal arrays to avoid kernel issues
            return (
                np.zeros((1, n_types, n_types), dtype=np.float32),
                np.zeros((1, 8), dtype=np.float32),
            )
        
        matrices = np.zeros((n_rules, n_types, n_types), dtype=np.float32)
        params = np.zeros((n_rules, 8), dtype=np.float32)
        
        for i, r in enumerate(active_rules):
            matrices[i] = r.matrix
            params[i, 0] = r.min_radius
            params[i, 1] = r.max_radius
            params[i, 2] = r.strength
            params[i, 3] = r.softening
            params[i, 4] = float(r.force_type)
            # Extra params (5, 6, 7) for advanced force types
            # Default to 0.0, can be extended per-rule
            params[i, 5] = 0.0  # param1 (e.g., decay_length, sigma)
            params[i, 6] = 0.0  # param2 (e.g., r0, well_width)
            params[i, 7] = 0.0  # param3
            
        return matrices, params

    def _pack_species(self):
        """Pack species config for kernel (6 columns now for spin dynamics)."""
        n = self.cfg.num_types
        arr = np.zeros((n, 6), dtype=np.float32)
        arr[:, 0] = self.species_config.radius
        arr[:, 1] = self.species_config.wave_freq
        arr[:, 2] = self.species_config.wave_amp
        arr[:, 3] = self.species_config.spin_inertia
        arr[:, 4] = self.species_config.spin_friction
        arr[:, 5] = self.species_config.base_spin_rate
        return arr

    def update(self, dt: float = None):
        """
        Velocity Verlet integration step with proper thermostat ordering.
        
        Includes:
        - Standard particle life forces
        - Exclusion mechanics (Pauli-like repulsion)
        - Spin flip dynamics
        - Thermostat AFTER velocity update (correct NVT ordering)
        """
        if dt is None: dt = self.cfg.dt
        n = self.state.active
        if n == 0:
            return
        
        self._frame += 1
        
        # World bounds
        half = self.cfg.half_world
        bounds = np.array([-half, half], dtype=np.float32)
        
        # Initialize force cache on first frame
        if not hasattr(self, 'forces_cache'):
            self.forces_cache = np.zeros_like(self.state.pos)
            self.torques_cache = np.zeros_like(self.state.angle)
            # Compute initial forces
            self._rebuild_grid()
            self._compute_all_forces()

        # Run substeps for stability
        # Use a FIXED internal dt for physics consistency regardless of frame rate
        # This ensures simulation behaves the same at any FPS
        base_dt = 0.005  # Fixed physics timestep
        sub_dt = base_dt / self.cfg.substeps
        
        # Number of physics steps to take this frame
        n_steps = max(1, int(dt / base_dt + 0.5))
        
        for _ in range(n_steps):
            for _ in range(self.cfg.substeps):
                # 1. First half of Velocity Verlet (with wall collision damping)
                integrate_verlet_1(
                    self.state.pos, self.state.vel, 
                    self.state.angle, self.state.ang_vel,
                    self.forces_cache, self.torques_cache, 
                    n, sub_dt, bounds,
                    self.cfg.collision_damping  # Haskell-style wall bounce damping
                )
                
                # 2. Compute forces at new positions r(t+dt)
                self._rebuild_grid()
                self._compute_all_forces()
                
                # 3. Second half of Velocity Verlet
                integrate_verlet_2(
                    self.state.vel, self.state.ang_vel,
                    self.forces_cache, self.torques_cache,
                    n, sub_dt, self.cfg.friction, self.cfg.angular_friction,
                    self.cfg.max_velocity,
                    1.0  # No per-substep slowdown
                )
            
            # 4. Haskell-style slowdown: v *= factor per physics step (not per substep!)
            # This is critical for correct emergence dynamics
            # The Haskell code applies this AFTER adding acceleration to velocity
            if self.cfg.slowdown_factor < 1.0:
                self.state.vel[:n] *= self.cfg.slowdown_factor
        
        # 5. Apply thermostat AFTER full velocity update (correct NVT ordering)
        if self.cfg.thermostat_enabled:
            apply_thermostat(
                self.state.vel, n, 
                self.cfg.target_temperature, 
                self.cfg.thermostat_coupling, 
                dt
            )
        
        # 5. Spin flip dynamics (stochastic, every few frames)
        if self.spin_flip_enabled and self._frame % 5 == 0:
            self._apply_spin_flips()
    
    def _compute_all_forces(self):
        """Compute all forces: standard + wave-perimeter exclusion + spin coupling."""
        n = self.state.active
        matrices, params = self._pack_rules()
        species_params = self._pack_species()
        
        # Standard particle life forces
        self.forces_cache, self.torques_cache = compute_forces_multi(
            self.state.pos, self.state.colors, self.state.angle, n,
            matrices, params, species_params,
            self.cfg.wave_repulsion_strength, self.cfg.wave_repulsion_exp,
            self.grid_counts, self.grid_cells, self.cell_size,
            self.cfg.gravity, self.cfg.half_world
        )
        
        # Add WAVE-PERIMETER exclusion forces if enabled
        if self.exclusion_enabled:
            apply_exclusion_forces_wave(
                self.state.pos, self.state.colors, self.state.angle, self.spin,
                species_params,  # Contains [radius, freq, amp, inertia, spin_fric, base_spin]
                self.behavior_matrix,
                self.forces_cache, self.torques_cache, n,
                self.exclusion_strength, self.exclusion_radius_factor,
                self.grid_counts, self.grid_cells, self.cell_size,
                self.cfg.half_world
            )
        
        # Add spin-spin coupling for angular dynamics
        if self.spin_enabled and hasattr(self, 'spin_coupling_matrix'):
            apply_spin_coupling(
                self.state.pos, self.spin, self.state.ang_vel, self.state.colors,
                self.spin_coupling_matrix, self.torques_cache, n,
                self.spin_coupling_strength, self.spin_interaction_range,
                self.grid_counts, self.grid_cells, self.cell_size,
                self.cfg.half_world
            )
    
    def _apply_spin_flips(self):
        """Apply stochastic spin flips based on particle energy."""
        n = self.state.active
        n_types = self.cfg.num_types
        
        # Build per-type arrays
        flip_threshold = np.full(n_types, self.spin_flip_threshold, dtype=np.float32)
        flip_probability = np.full(n_types, self.spin_flip_probability, dtype=np.float32)
        spin_enabled = np.ones(n_types, dtype=np.bool_)
        
        apply_spin_flip(
            self.state.vel, self.spin, self.state.colors,
            flip_threshold, flip_probability, spin_enabled,
            n, self._frame
        )
    
    def get_spin_statistics(self):
        """Get current spin distribution statistics."""
        return compute_spin_statistics(
            self.spin, self.state.pos, self.state.active,
            correlation_range=self.cfg.world_size * 0.1
        )
    
    def _rebuild_grid(self):
        """Rebuild spatial grid with overflow detection."""
        n = self.state.active
        fill_grid(self.state.pos, n, self.cell_size, self.grid_counts, self.grid_cells, self.cfg.half_world)
        # Check for overflow
        max_count = np.max(self.grid_counts)
        if max_count >= self.max_per_cell:
            # Dynamically resize grid capacity
            self._resize_grid(max_count * 2)
    
    def _resize_grid(self, new_max_per_cell: int):
        """Resize grid cell capacity to handle denser regions."""
        self.max_per_cell = new_max_per_cell
        self.grid_cells = np.zeros((self.grid_h, self.grid_w, self.max_per_cell), dtype=np.int32)
        # Refill with new capacity
        fill_grid(self.state.pos, self.state.active, self.cell_size, self.grid_counts, self.grid_cells, self.cfg.half_world)
