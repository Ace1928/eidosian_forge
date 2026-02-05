"""
Physics Engine Manager.
Handles memory allocation, state management, and multi-rule configuration.
"""
import numpy as np
from typing import List
from ..core.types import ParticleState, SimulationConfig, InteractionRule, ForceType
from .kernels import fill_grid, compute_forces_multi, integrate

class PhysicsEngine:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        
        # State Container
        self.state = ParticleState.allocate(config.max_particles)
        self.state.active = config.num_particles
        
        # Initialize Rules
        self.rules: List[InteractionRule] = []
        self._init_default_rules()
        
        # Initialize Random Particles
        self._init_particles()
        
        # Grid Memory
        # Must be large enough for the largest interaction radius
        self.max_interaction_radius = self._get_max_radius()
        self.cell_size = self.max_interaction_radius if self.max_interaction_radius > 0 else 0.1
        self.grid_w = int(2.0 / self.cell_size) + 2
        self.grid_h = int(2.0 / self.cell_size) + 2
        
        avg_density = config.num_particles / (self.grid_w * self.grid_h)
        self.max_per_cell = int(avg_density * 20) + 100 
        
        self.grid_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.grid_cells = np.zeros((self.grid_h, self.grid_w, self.max_per_cell), dtype=np.int32)
        
    def _init_default_rules(self):
        """Create the default Linear and Gravity rules."""
        # 1. Standard Linear (Particle Life)
        mat_linear = np.random.uniform(-1.0, 1.0, (self.cfg.num_types, self.cfg.num_types)).astype(np.float32)
        rule_lin = InteractionRule(
            name="Particle Life (Linear)",
            force_type=ForceType.LINEAR,
            matrix=mat_linear,
            max_radius=self.cfg.default_max_radius,
            min_radius=self.cfg.default_min_radius,
            strength=1.0
        )
        self.rules.append(rule_lin)
        
        # 2. Gravity / Inverse Square
        # Default mostly zero, user can enable it
        mat_grav = np.zeros((self.cfg.num_types, self.cfg.num_types), dtype=np.float32)
        rule_grav = InteractionRule(
            name="Gravity (InvSq)",
            force_type=ForceType.INVERSE_SQUARE,
            matrix=mat_grav,
            max_radius=0.5, # Larger range
            min_radius=0.01,
            strength=0.5,
            softening=0.05
        )
        self.rules.append(rule_grav)

    def _get_max_radius(self):
        if not self.rules: return 0.1
        return max(r.max_radius for r in self.rules)

    def _init_particles(self):
        n = self.state.active
        self.state.pos[:n] = np.random.uniform(-1.0, 1.0, (n, 2))
        self.state.vel[:n] = 0.0
        self.state.colors[:n] = np.random.randint(0, self.cfg.num_types, n)

    def reset(self):
        self._init_particles()

    def set_active_count(self, count: int):
        if count > self.cfg.max_particles: count = self.cfg.max_particles
        old_count = self.state.active
        self.state.active = count
        if count > old_count:
            diff = count - old_count
            self.state.pos[old_count:count] = np.random.uniform(-1.0, 1.0, (diff, 2))
            self.state.vel[old_count:count] = 0.0
            self.state.colors[old_count:count] = np.random.randint(0, self.cfg.num_types, diff)

    def _pack_rules(self):
        """
        Pack rules into Numba-friendly arrays.
        """
        n_rules = len(self.rules)
        n_types = self.cfg.num_types
        
        # Matrices: (N, T, T)
        matrices = np.zeros((n_rules, n_types, n_types), dtype=np.float32)
        
        # Params: (N, 5) -> [min_r, max_r, strength, softening, type]
        params = np.zeros((n_rules, 5), dtype=np.float32)
        
        for i, r in enumerate(self.rules):
            matrices[i] = r.matrix
            params[i, 0] = r.min_radius
            params[i, 1] = r.max_radius
            params[i, 2] = r.strength
            params[i, 3] = r.softening
            params[i, 4] = float(r.force_type)
            
        return matrices, params

    def update(self, dt: float = None):
        if dt is None: dt = self.cfg.dt
        n = self.state.active
        
        # Check if max radius changed (e.g. via UI) and rebuild grid size if needed
        current_max_r = self._get_max_radius()
        if abs(current_max_r - self.cell_size) > 0.01:
            # Re-alloc grid if radius grew significantly
            # For simplicity, we just use current cell_size unless it's too small
            if current_max_r > self.cell_size:
                self.cell_size = current_max_r
                # Re-alloc logic omitted for performance stability in this iteration,
                # assuming config ranges don't explode.
        
        # 1. Spatial Hash
        fill_grid(
            self.state.pos, n, self.cell_size,
            self.grid_counts, self.grid_cells
        )
        
        # 2. Compute Forces (Multi-Rule)
        matrices, params = self._pack_rules()
        
        forces = compute_forces_multi(
            self.state.pos, self.state.colors, n,
            matrices, params,
            self.grid_counts, self.grid_cells, self.cell_size,
            dt, self.cfg.friction, self.cfg.gravity
        )
        
        # 3. Integrate
        integrate(
            self.state.pos, self.state.vel, forces, n,
            dt, self.cfg.friction, np.array([-1.0, 1.0])
        )
