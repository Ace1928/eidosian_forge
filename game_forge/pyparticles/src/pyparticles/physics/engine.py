"""
Physics Engine Manager.
Updated for Velocity Verlet Integration.
"""
import numpy as np
from typing import List
from ..core.types import ParticleState, SimulationConfig, InteractionRule, ForceType, SpeciesConfig
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
        
        # Species Config
        self.species_config = SpeciesConfig.default(config.num_types)
        
        # Initialize Random Particles
        self._init_particles()
        
        # Grid Memory
        self.max_interaction_radius = self._get_max_radius()
        self.cell_size = max(self.max_interaction_radius, 0.2)
        
        self.grid_w = int(2.0 / self.cell_size) + 2
        self.grid_h = int(2.0 / self.cell_size) + 2
        
        avg_density = config.num_particles / (self.grid_w * self.grid_h)
        self.max_per_cell = int(avg_density * 20) + 100 
        
        self.grid_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.grid_cells = np.zeros((self.grid_h, self.grid_w, self.max_per_cell), dtype=np.int32)
        
    def _init_default_rules(self):
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
        
        mat_grav = np.zeros((self.cfg.num_types, self.cfg.num_types), dtype=np.float32)
        rule_grav = InteractionRule(
            name="Gravity (InvSq)",
            force_type=ForceType.INVERSE_SQUARE,
            matrix=mat_grav,
            max_radius=0.5, 
            min_radius=0.01,
            strength=0.5,
            softening=0.05
        )
        self.rules.append(rule_grav)
        
        mat_strong = np.zeros((self.cfg.num_types, self.cfg.num_types), dtype=np.float32)
        rule_strong = InteractionRule(
            name="Strong Force (InvCube)",
            force_type=ForceType.INVERSE_CUBE,
            matrix=mat_strong,
            max_radius=0.2, 
            min_radius=0.01,
            strength=2.0,
            softening=0.02
        )
        self.rules.append(rule_strong)

    def _get_max_radius(self):
        if not self.rules: return 0.1
        return max(r.max_radius for r in self.rules)

    def _init_particles(self):
        n = self.state.active
        self.state.pos[:n] = np.random.uniform(-1.0, 1.0, (n, 2))
        self.state.vel[:n] = 0.0
        self.state.colors[:n] = np.random.randint(0, self.cfg.num_types, n)
        self.state.angle[:n] = np.random.uniform(0, 2*np.pi, n)
        
        for i in range(n):
            t = self.state.colors[i]
            self.state.ang_vel[i] = self.species_config.wave_phase_speed[t]

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
            self.state.angle[old_count:count] = np.random.uniform(0, 2*np.pi, diff)
            for i in range(old_count, count):
                t = self.state.colors[i]
                self.state.ang_vel[i] = self.species_config.wave_phase_speed[t]

    def set_species_count(self, n_types: int):
        if n_types == self.cfg.num_types: return
        self.cfg.num_types = n_types
        self.species_config = SpeciesConfig.default(n_types)
        self.rules = []
        self._init_default_rules()
        self.reset()

    def _pack_rules(self):
        active_rules = [r for r in self.rules if r.enabled]
        n_rules = len(active_rules)
        n_types = self.cfg.num_types
        
        matrices = np.zeros((n_rules, n_types, n_types), dtype=np.float32)
        params = np.zeros((n_rules, 5), dtype=np.float32)
        
        for i, r in enumerate(active_rules):
            matrices[i] = r.matrix
            params[i, 0] = r.min_radius
            params[i, 1] = r.max_radius
            params[i, 2] = r.strength
            params[i, 3] = r.softening
            params[i, 4] = float(r.force_type)
            
        return matrices, params

    def _pack_species(self):
        n = self.cfg.num_types
        arr = np.zeros((n, 3), dtype=np.float32)
        arr[:, 0] = self.species_config.radius
        arr[:, 1] = self.species_config.wave_freq
        arr[:, 2] = self.species_config.wave_amp
        return arr

    def update(self, dt: float = None):
        if dt is None: dt = self.cfg.dt
        n = self.state.active
        
        # 0. Thermostat (Check Temp & Scale Velocity)
        # Apply BEFORE integration step to regulate KE?
        # Or after full step?
        # Standard: Every step or every N steps.
        apply_thermostat(
            self.state.vel, n, 
            self.cfg.target_temperature, 
            self.cfg.thermostat_coupling, 
            dt
        )
        
        # 1. First Integration Step (Velocity Verlet Part 1)
        # r(t+dt), v(t+0.5dt) using forces(t)
        # BUT we haven't computed forces(t) for the VERY FIRST FRAME if we just started.
        # We need stored forces from previous step.
        # Initialize forces if needed?
        if not hasattr(self, 'forces_cache'):
            self.forces_cache = np.zeros_like(self.state.pos)
            self.torques_cache = np.zeros_like(self.state.angle)
            # Initial Force Calc
            fill_grid(self.state.pos, n, self.cell_size, self.grid_counts, self.grid_cells)
            matrices, params = self._pack_rules()
            species_params = self._pack_species()
            self.forces_cache, self.torques_cache = compute_forces_multi(
                self.state.pos, self.state.colors, self.state.angle, n,
                matrices, params, species_params,
                self.cfg.wave_repulsion_strength, self.cfg.wave_repulsion_exp,
                self.grid_counts, self.grid_cells, self.cell_size,
                self.cfg.gravity
            )

        integrate_verlet_1(
            self.state.pos, self.state.vel, 
            self.state.angle, self.state.ang_vel,
            self.forces_cache, self.torques_cache, 
            n, dt, np.array([-1.0, 1.0])
        )
        
        # 2. Compute Forces at New Position r(t+dt)
        fill_grid(self.state.pos, n, self.cell_size, self.grid_counts, self.grid_cells)
        matrices, params = self._pack_rules()
        species_params = self._pack_species()
        
        new_forces, new_torques = compute_forces_multi(
            self.state.pos, self.state.colors, self.state.angle, n,
            matrices, params, species_params,
            self.cfg.wave_repulsion_strength, self.cfg.wave_repulsion_exp,
            self.grid_counts, self.grid_cells, self.cell_size,
            self.cfg.gravity
        )
        
        # 3. Second Integration Step (Velocity Verlet Part 2)
        integrate_verlet_2(
            self.state.vel, self.state.ang_vel,
            new_forces, new_torques,
            n, dt, self.cfg.friction
        )
        
        # Update Cache
        self.forces_cache = new_forces
        self.torques_cache = new_torques
