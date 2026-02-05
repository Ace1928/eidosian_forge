"""
Physics Engine Manager.
Updated for Velocity Verlet Integration with configurable world size.
"""
import numpy as np
from typing import List
from ..core.types import (
    ParticleState, SimulationConfig, InteractionRule, ForceType, 
    SpeciesConfig, SpinInteractionMatrix
)
from .kernels import (
    fill_grid, compute_forces_multi, apply_thermostat, 
    integrate_verlet_1, integrate_verlet_2
)

class PhysicsEngine:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        
        # State Container
        self.state = ParticleState.allocate(config.max_particles)
        self.state.active = config.num_particles
        
        # Initialize Rules
        self.rules: List[InteractionRule] = []
        self._init_default_rules()
        
        # Species Config - scaled to world size
        self.species_config = SpeciesConfig.default(config.num_types, config.world_size)
        
        # Spin interaction matrix
        self.spin_matrix = SpinInteractionMatrix.default(config.num_types)
        
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

    def _get_max_radius(self):
        if not self.rules: 
            return self.cfg.default_max_radius
        return max(r.max_radius for r in self.rules)

    def _init_particles(self):
        """Initialize particles with species-driven properties."""
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
        self._invalidate_cache()

    def set_species_count(self, n_types: int):
        if n_types == self.cfg.num_types: return
        self.cfg.num_types = n_types
        self.species_config = SpeciesConfig.default(n_types, self.cfg.world_size)
        self.spin_matrix = SpinInteractionMatrix.default(n_types)
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
        
        Correct NVT ensemble: thermostat AFTER velocity update, not before.
        This ensures we measure actual kinetic energy before scaling.
        """
        if dt is None: dt = self.cfg.dt
        n = self.state.active
        if n == 0:
            return
        
        # World bounds
        half = self.cfg.half_world
        bounds = np.array([-half, half], dtype=np.float32)
        
        # Initialize force cache on first frame
        if not hasattr(self, 'forces_cache'):
            self.forces_cache = np.zeros_like(self.state.pos)
            self.torques_cache = np.zeros_like(self.state.angle)
            # Compute initial forces
            self._rebuild_grid()
            matrices, params = self._pack_rules()
            species_params = self._pack_species()
            self.forces_cache, self.torques_cache = compute_forces_multi(
                self.state.pos, self.state.colors, self.state.angle, n,
                matrices, params, species_params,
                self.cfg.wave_repulsion_strength, self.cfg.wave_repulsion_exp,
                self.grid_counts, self.grid_cells, self.cell_size,
                self.cfg.gravity, half
            )

        # Run substeps for stability
        sub_dt = dt / self.cfg.substeps
        for _ in range(self.cfg.substeps):
            # 1. First half of Velocity Verlet
            integrate_verlet_1(
                self.state.pos, self.state.vel, 
                self.state.angle, self.state.ang_vel,
                self.forces_cache, self.torques_cache, 
                n, sub_dt, bounds
            )
            
            # 2. Compute forces at new positions r(t+dt)
            self._rebuild_grid()
            matrices, params = self._pack_rules()
            species_params = self._pack_species()
            
            new_forces, new_torques = compute_forces_multi(
                self.state.pos, self.state.colors, self.state.angle, n,
                matrices, params, species_params,
                self.cfg.wave_repulsion_strength, self.cfg.wave_repulsion_exp,
                self.grid_counts, self.grid_cells, self.cell_size,
                self.cfg.gravity, half
            )
            
            # 3. Second half of Velocity Verlet with velocity capping
            integrate_verlet_2(
                self.state.vel, self.state.ang_vel,
                new_forces, new_torques,
                n, sub_dt, self.cfg.friction, self.cfg.angular_friction,
                self.cfg.max_velocity
            )
            
            # Update force cache for next substep
            self.forces_cache = new_forces
            self.torques_cache = new_torques
        
        # 4. Apply thermostat AFTER full velocity update (correct NVT ordering)
        if self.cfg.thermostat_enabled:
            apply_thermostat(
                self.state.vel, n, 
                self.cfg.target_temperature, 
                self.cfg.thermostat_coupling, 
                dt
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
