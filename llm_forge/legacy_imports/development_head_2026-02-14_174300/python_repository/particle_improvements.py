class CellularTypeData:
    """
    Represents a cellular type with multiple cellular components.
    Manages positions, velocities, energy, mass, and genetic traits of components.
    """

    def __init__(self,
                 type_id: int,
                 color: Tuple[int, int, int],
                 n_particles: int,
                 window_width: int,
                 window_height: int,
                 initial_energy: float,
                 max_age: float = np.inf,
                 mass: Optional[float] = None,
                 base_velocity_scale: float = 1.0,
                 energy_efficiency: Optional[float] = None,
                 gene_traits: List[str] = ["speed_factor", "interaction_strength",
                                           "perception_range", "reproduction_rate",
                                           "synergy_affinity", "colony_factor",
                                           "drift_sensitivity", "max_energy_storage",
                                           "sensory_sensitivity", "memory_transfer_rate",
                                           "communication_range", "socialization_tendency",
                                           "colony_building_skill", "cultural_influence"],
                 gene_mutation_rate: float = 0.05,
                 gene_mutation_range: Tuple[float, float] = (-0.1, 0.1),
                 min_energy: float = 0.0,
                 max_energy: float = 1000.0,
                 min_mass: float = 0.1,
                 max_mass: float = 10.0,
                 min_velocity: float = -10.0,
                 max_velocity: float = 10.0,
                 min_perception: float = 10.0,
                 max_perception: float = 300.0,
                 min_reproduction: float = 0.05,
                 max_reproduction: float = 1.0,
                 min_synergy: float = 0.0,
                 max_synergy: float = 2.0,
                 min_colony: float = 0.0,
                 max_colony: float = 1.0,
                 min_drift: float = 0.0,
                 max_drift: float = 2.0,
                 min_energy_efficiency: float = -0.3,
                 max_energy_efficiency: float = 2.5,
                 min_memory: float = 0.0,
                 max_memory: float = 1.0,
                 min_social: float = 0.0,
                 max_social: float = 1.0,
                 min_colony_build: float = 0.0,
                 max_colony_build: float = 1.0,
                 min_culture: float = 0.0,
                 max_culture: float = 1.0):
        """
        Initialize a CellularTypeData instance with given parameters.
        """
        # ... (Input validation and metadata storage remain the same) ...

        # Use a dictionary to store parameter bounds
        self.bounds = {
            'energy': (float(min_energy), float(max_energy)),
            'mass': (float(min_mass), float(max_mass)),
            'velocity': (float(min_velocity), float(max_velocity)),
            'perception': (float(min_perception), float(max_perception)),
            'reproduction': (float(min_reproduction), float(max_reproduction)),
            'synergy': (float(min_synergy), float(max_synergy)),
            'colony': (float(min_colony), float(max_colony)),
            'drift': (float(min_drift), float(max_drift)),
            'energy_efficiency': (float(min_energy_efficiency), float(max_energy_efficiency)),
            'memory': (float(min_memory), float(max_memory)),
            'social': (float(min_social), float(max_social)),
            'colony_build': (float(min_colony_build), float(max_colony_build)),
            'culture': (float(min_culture), float(max_culture))
        }

        try:
            # Initialize arrays using helper method
            self._initialize_arrays(n_particles, window_width, window_height,
                                   initial_energy, base_velocity_scale, energy_efficiency)

            # ... (Initialization of gene traits remains the same) ...

            # ... (Initialization of tracking arrays remains the same) ...

            # ... (Store mutation parameters remains the same) ...

        except Exception as e:
            # ... (Fallback initialization remains the same) ...

    def _initialize_arrays(self, n_particles, window_width, window_height,
                           initial_energy, base_velocity_scale, energy_efficiency):
        """Helper method to initialize arrays with safe values."""
        # Initialize positions safely
        coords = random_xy(window_width, window_height, n_particles)
        self.x = coords[:, 0].astype(np.float64)
        self.y = coords[:, 1].astype(np.float64)

        # Initialize energy efficiency safely
        if energy_efficiency is None:
            self.energy_efficiency = np.clip(
                np.random.uniform(self.bounds['energy_efficiency'][0],
                                 self.bounds['energy_efficiency'][1], n_particles),
                *self.bounds['energy_efficiency']
            ).astype(np.float64)
        else:
            self.energy_efficiency = np.full(n_particles, np.clip(
                float(energy_efficiency),
                *self.bounds['energy_efficiency']
            ), dtype=np.float64)

        # ... (Safe velocity and energy initialization remain the same) ...

        # Initialize mass safely for mass-based types
        if self.mass_based:
            mass_value = mass if mass is not None and mass > 0.0 else self.bounds['mass'][0]
            self.mass = np.full(n_particles, mass_value, dtype=np.float64)
        else:
            self.mass = None

        # ... (Initialization of status arrays remains the same) ...

    def _validate_array_shapes(self) -> None:
        """Validate and correct array shapes for consistency."""
        # ... (Implementation remains the same) ...

    def is_alive_mask(self) -> np.ndarray:
        """Compute alive mask with safe array operations."""
        # ... (Implementation remains the same) ...

    def update_alive(self) -> None:
        """Update alive status safely."""
        # ... (Implementation remains the same) ...

    def age_components(self) -> None:
        """Age components with safe operations."""
        # ... (Implementation remains the same) ...

    def update_states(self) -> None:
        """Update component states safely."""
        # ... (Implementation remains the same) ...

    def remove_dead(self, config: SimulationConfig) -> None:
        """Remove dead components safely with array broadcasting."""
        # ... (Implementation remains the same) ...

    def _handle_energy_transfer(self, dead_due_to_age: np.ndarray, alive_mask: np.ndarray, config: SimulationConfig) -> None:
        """Handle energy transfer from dead components safely."""
        # ... (Implementation remains the same) ...

    def add_component(self,
                      x: float,
                      y: float,
                      vx: float,
                      vy: float,
                      energy: float,
                      mass_val: Optional[float],
                      energy_efficiency_val: float,
                      speed_factor_val: float,
                      interaction_strength_val: float,
                      perception_range_val: float,
                      reproduction_rate_val: float,
                      synergy_affinity_val: float,
                      colony_factor_val: float,
                      drift_sensitivity_val: float,
                      species_id_val: int,
                      parent_id_val: int,
                      max_age: float,
                      max_energy_storage_val: float,
                      sensory_sensitivity_val: float,
                      memory_transfer_rate_val: float,
                      communication_range_val: float,
                      socialization_tendency_val: float,
                      colony_building_skill_val: float,
                      cultural_influence_val: float
                      ) -> None:
        """Add new component safely with array broadcasting."""
        # ... (Implementation remains the same) ...


###############################################################
# Genetic Instructions (Turing-Complete Neural Architecture)
###############################################################

class GeneticInstructions:
    """Advanced Turing-complete neural instruction architecture with optimized execution."""

    # ... (Instruction set and parameters remain the same) ...

    def __init__(self):
        """Initialize the neural instruction architecture."""
        # ... (Core component initialization remains the same) ...

        # Neural components are now initialized using a helper method
        self._initialize_neural_components()

        # ... (Optimization components remain the same) ...

    def _initialize_neural_components(self):
        """Helper method to initialize neural network components."""
        self.weights = {}
        self.biases = {}
        self.gradients = {}
        self.activations = {}

    def _setup_execution_pipeline(self):
        """Setup optimized execution pipeline."""
        # ... (Pipeline and dispatch table remain the same) ...

        # Vectorized operations are now accessed through a helper method
        self.vector_ops = self._get_vectorized_operations()

    def _get_vectorized_operations(self):
        """Helper method to return a dictionary of vectorized operations."""
        return {
            'ADD': np.add,
            'MUL': np.multiply,
            'DIV': np.divide,
            'POW': np.power,
            'SQRT': np.sqrt,
            'LOG': np.log,
            'EXP': np.exp
        }

    def _init_neural_functions(self):
        """Initialize neural network activation functions."""
        # ... (Activation and gradient functions remain the same) ...

    @staticmethod
    def create_optimized_sequence(length: int,
                                  instruction_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Create an optimized instruction sequence with optional weighting."""
        # ... (Implementation remains the same) ...

    # ... (Rest of the class implementation remains the same) ...


###############################################################
# Genome Class
###############################################################

class Genome:
    """
    Genome: sequence of instructions + regulatory info.
    """

    def __init__(self, config: SimulationConfig):
        # ... (Initialization remains the same) ...

    def _random_genome(self):
        """Generate a random genome sequence."""
        # ... (Implementation remains the same) ...

    def mutate(self):
        """Apply various mutation operators to the genome."""
        # ... (Implementation remains the same) ...


###############################################################
# Genetic Interpreter Class
###############################################################

class GeneticInterpreter:
    """Advanced genetic sequence interpreter implementing Turing-complete genetic programming."""

    def __init__(self, gene_sequence: Optional[List[List[Any]]] = None):
        """Initialize genetic interpreter with optimized defaults."""
        # ... (Default sequence and bounds remain the same) ...

        # Initialize core components using helper methods
        self._initialize_genetic_mechanisms()
        self._setup_caches()

    def _initialize_genetic_mechanisms(self) -> None:
        """Initialize advanced genetic control mechanisms."""
        # ... (Regulatory networks, epistatic interactions, and epigenetic modifications remain the same) ...

    def _setup_caches(self) -> None:
        """Initialize performance optimization caches."""
        # ... (Implementation remains the same) ...

    def decode(self, particle: CellularTypeData, others: List[CellularTypeData], env: SimulationConfig) -> None:
        """Decode genetic sequence with comprehensive error handling."""
        # ... (Cache clearing and gene processing remain the same) ...

    def _check_regulatory_state(self, gene_type: str, particle: CellularTypeData) -> bool:
        """Check if gene expression is allowed by regulatory networks."""
        # ... (Implementation remains the same) ...

    def _apply_epistatic_effects(self, gene_type: str, gene_data: np.ndarray, particle: CellularTypeData) -> np.ndarray:
        """Apply epistatic interactions between genes."""
        # ... (Implementation remains the same) ...

    def _apply_epigenetic_mods(self, gene_type: str, gene_data: np.ndarray) -> np.ndarray:
        """Apply epigenetic modifications to gene expression."""
        # ... (Implementation remains the same) ...

    def _ensure_particle_stability(self, particle: CellularTypeData) -> None:
        """Ensure particle maintains valid state after operations."""
        # ... (Implementation remains the same) ...

    # ... (Rest of the class implementation remains the same) ...