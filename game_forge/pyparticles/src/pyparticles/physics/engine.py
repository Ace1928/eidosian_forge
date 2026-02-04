"""
Physics Engine Manager.
Handles memory allocation, state management, and grid sizing.
"""
import numpy as np
from ..core.types import ParticleState, SimulationConfig
from .kernels import fill_grid, compute_forces, integrate

class PhysicsEngine:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        
        # State Container
        self.state = ParticleState.allocate(config.max_particles)
        self.state.active = config.num_particles
        
        # Interaction Matrix
        self.matrix = np.random.uniform(-1.0, 1.0, (config.num_types, config.num_types)).astype(np.float32)
        
        # Initialize Random Particles
        self._init_particles()
        
        # Grid Memory
        self.cell_size = config.max_radius
        self.grid_w = int(2.0 / self.cell_size) + 2
        self.grid_h = int(2.0 / self.cell_size) + 2
        
        # Heuristic for max particles per cell
        avg_density = config.num_particles / (self.grid_w * self.grid_h)
        self.max_per_cell = int(avg_density * 20) + 100 
        
        self.grid_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.grid_cells = np.zeros((self.grid_h, self.grid_w, self.max_per_cell), dtype=np.int32)
        
    def _init_particles(self):
        n = self.state.active
        self.state.pos[:n] = np.random.uniform(-1.0, 1.0, (n, 2))
        self.state.vel[:n] = 0.0
        self.state.colors[:n] = np.random.randint(0, self.cfg.num_types, n)

    def reset(self):
        """Reset positions and velocities."""
        self._init_particles()

    def set_active_count(self, count: int):
        """Update active particle count dynamically."""
        if count > self.cfg.max_particles:
            count = self.cfg.max_particles
        
        old_count = self.state.active
        self.state.active = count
        
        # If increased, initialize new ones
        if count > old_count:
            diff = count - old_count
            self.state.pos[old_count:count] = np.random.uniform(-1.0, 1.0, (diff, 2))
            self.state.vel[old_count:count] = 0.0
            self.state.colors[old_count:count] = np.random.randint(0, self.cfg.num_types, diff)

    def update(self, dt: float = None):
        """Main Physics Step."""
        if dt is None:
            dt = self.cfg.dt
            
        n = self.state.active
        
        # 1. Spatial Hash
        fill_grid(
            self.state.pos, n, self.cell_size,
            self.grid_counts, self.grid_cells
        )
        
        # 2. Compute Forces
        forces = compute_forces(
            self.state.pos, self.state.colors, n, self.matrix,
            self.grid_counts, self.grid_cells, self.cell_size,
            dt, self.cfg.friction, self.cfg.max_radius, 
            self.cfg.min_radius, self.cfg.repulsion_strength, self.cfg.gravity
        )
        
        # 3. Integrate
        integrate(
            self.state.pos, self.state.vel, forces, n,
            dt, self.cfg.friction, np.array([-1.0, 1.0])
        )
