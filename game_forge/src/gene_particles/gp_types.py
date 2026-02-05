"""Gene Particles Type System Module.

Provides comprehensive type definitions and data structures for the Gene Particles simulation,
including genetic trait architecture, cellular component management, evolutionary mechanics,
and emergent behavior algorithms with rigorous type safety.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from eidosian_core import eidosian
from game_forge.src.gene_particles.gp_utility import (
    tile_positions_for_wrap,
    wrap_deltas,
    wrap_positions,
)

# Import scipy's cKDTree with explicit type handling
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore[import]
except ImportError:
    # Fallback for type checking
    class KDTree:
        """Type stub for SciPy's KDTree class when imports fail."""

        def __init__(self, data: NDArray[np.float64], leafsize: int = 10):
            pass

        @eidosian()
        def query(
            self,
            x: Any,
            k: int = 1,
            eps: float = 0.0,
            p: float = 2.0,
            distance_upper_bound: float = np.inf,
            workers: Optional[int] = 1,
        ) -> Tuple[Any, Any]:
            """Stub for query method."""
            return np.array([]), np.array([])


# Use TYPE_CHECKING to break circular import
if TYPE_CHECKING:
    from game_forge.src.gene_particles.gp_config import SimulationConfig

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Type Definitions: Precision before the first keystroke                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Domain-specific vector and data types with precise semantics
Vector2D = Tuple[float, float]  # (x, y) coordinate pair in Cartesian space
Vector3D = Tuple[float, float, float]  # (x, y, z) coordinate tuple in Cartesian space
ColorRGB = Tuple[int, int, int]  # (r, g, b) color values constrained to 0-255
Range = Tuple[float, float]  # (min, max) value constraints for trait boundaries

# NumPy array specialized types for simulation operations
FloatArray = NDArray[np.float64]  # For continuous genetic traits and positions
IntArray = NDArray[np.int64]  # For discrete identifiers and counters
BoolArray = NDArray[np.bool_]  # For vectorized condition masks

# Generic type variable for type-preserving functions
T = TypeVar("T")  # Enables generic functions that preserve input/output types

# Validation result typing with explicit semantics
ValidationResult = Union[Literal[True], str]  # Success or error message
VALID: Final[Literal[True]] = True  # Canonical success value

# Constraint definitions
RangeConstraint = Dict[str, Range]  # Parameter name to valid range mapping
PROBABILITY_BOUNDS: Final[Range] = (0.0, 1.0)  # For probability parameters

# Type aliases for better readability
GeneData = List[float]
GeneSequence = List[List[Union[str, float]]]
TraitValue = Union[
    float, NDArray[np.float64]
]  # Allow both scalar and array trait values


class GeneType(Enum):
    """Precise categorization of genetic element functions."""

    MOVEMENT = auto()
    INTERACTION = auto()
    ENERGY = auto()
    REPRODUCTION = auto()
    GROWTH = auto()
    PREDATION = auto()


# Gene type mapping for enhanced performance and type safety
GENE_TYPE_MAP: Dict[str, GeneType] = {
    "start_movement": GeneType.MOVEMENT,
    "start_interaction": GeneType.INTERACTION,
    "start_energy": GeneType.ENERGY,
    "start_reproduction": GeneType.REPRODUCTION,
    "start_growth": GeneType.GROWTH,
    "start_predation": GeneType.PREDATION,
}


class TraitType(Enum):
    """
    Genetic trait categories for organizational and validation purposes.

    Each category represents a distinct aspect of particle behavior that
    can be genetically influenced and mutated during evolution.
    """

    MOVEMENT = auto()  # Traits affecting particle motion and velocity
    INTERACTION = auto()  # Traits affecting inter-particle forces and reactions
    PERCEPTION = auto()  # Traits affecting sensing distance and environmental awareness
    REPRODUCTION = (
        auto()
    )  # Traits affecting breeding frequency and offspring characteristics
    SOCIAL = auto()  # Traits affecting group formation and collective behaviors
    ADAPTATION = (
        auto()
    )  # Traits affecting evolutionary plasticity and response to pressure


@dataclass(frozen=True)
class TraitDefinition:
    """
    Immutable definition of a genetic trait's properties and constraints.

    Acts as a schema for a genetic trait, defining its valid range,
    classification, and baseline values. Immutability ensures trait
    definitions remain consistent throughout simulation runtime.

    Attributes:
        name: Unique identifier for the trait
        type: Categorical classification of trait purpose
        range: Valid minimum/maximum values as (min, max) tuple
        description: Human-readable explanation of trait function
        default: Starting value for initialization
    """

    name: str
    type: TraitType
    range: Range
    description: str
    default: float

    def __post_init__(self) -> None:
        """Validate trait definition parameters upon initialization."""
        if not self.name:
            raise ValueError("Trait name must be a non-empty string")

        if len(self.range) != 2:
            raise ValueError("Range must be a tuple of two numeric values")

        if self.range[0] >= self.range[1]:
            raise ValueError(
                f"Range minimum ({self.range[0]}) must be less than maximum ({self.range[1]})"
            )

        if not self.description:
            raise ValueError("Description must be a non-empty string")

        if not (self.range[0] <= self.default <= self.range[1]):
            raise ValueError(
                f"Default value {self.default} is outside valid range {self.range}"
            )


class Validator(Protocol):
    """Protocol defining the validation interface for configuration objects."""

    def _validate(self) -> None:
        """
        Verify configuration integrity and parameter constraints.

        Raises:
            ValueError: If any parameter violates defined constraints
        """
        ...


class CellularTypeProtocol(Protocol):
    """Structural protocol defining the expected shape of cellular type data.

    Acts as a type contract for any data structure representing a cellular type,
    ensuring it provides all required attributes for rendering operations.
    """

    alive: NDArray[np.bool_]  # Boolean mask of living components
    x: NDArray[np.float64]  # X-coordinates
    y: NDArray[np.float64]  # Y-coordinates
    z: NDArray[np.float64]  # Z-coordinates (depth)
    energy: NDArray[np.float64]  # Energy levels
    speed_factor: NDArray[np.float64]  # Speed trait values
    color: ColorRGB  # RGB color tuple for this cellular type


# Helper function to generate random coordinates
@eidosian()
def random_xy(window_width: int, window_height: int, n: int = 1) -> "FloatArray":
    """Generate random position coordinates within window boundaries.

    Creates vectorized random positions for efficient particle initialization
    within the specified simulation window dimensions.

    Args:
        window_width: Width of simulation window in pixels
        window_height: Height of simulation window in pixels
        n: Number of coordinate pairs to generate

    Returns:
        FloatArray: Array of shape (n, 2) containing random (x, y) coordinates

    Raises:
        AssertionError: If window dimensions are non-positive or n < 1
    """
    assert window_width > 0, "Window width must be positive"
    assert window_height > 0, "Window height must be positive"
    assert n > 0, "Number of points must be positive"

    # Generate uniform random coordinates and explicitly cast to ensure type safety
    coords: "FloatArray" = np.random.uniform(
        0, [window_width, window_height], (n, 2)
    ).astype(np.float64)

    return coords


@eidosian()
def random_xyz(
    window_width: int, window_height: int, window_depth: int, n: int = 1
) -> "FloatArray":
    """Generate random 3D position coordinates within window boundaries.

    Creates vectorized random positions for efficient particle initialization
    within the specified simulation volume dimensions.

    Args:
        window_width: Width of simulation window in pixels
        window_height: Height of simulation window in pixels
        window_depth: Depth of simulation volume in pixels
        n: Number of coordinate triplets to generate

    Returns:
        FloatArray: Array of shape (n, 3) containing random (x, y, z) coordinates

    Raises:
        AssertionError: If window dimensions are non-positive or n < 1
    """
    assert window_width > 0, "Window width must be positive"
    assert window_height > 0, "Window height must be positive"
    assert window_depth > 0, "Window depth must be positive"
    assert n > 0, "Number of points must be positive"

    coords: "FloatArray" = np.random.uniform(
        0, [window_width, window_height, window_depth], (n, 3)
    ).astype(np.float64)

    return coords


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Type Definitions: Precise interaction parameter schemas                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class InteractionType(Enum):
    """Classification of physical interaction types between cellular entities.

    Defines the fundamental force models used for particle interactions:
        - POTENTIAL: Distance-based radial forces (attraction/repulsion)
        - GRAVITY: Mass-based gravitational forces (always attractive)
        - HYBRID: Combined potential and gravity (complex behaviors)
    """

    POTENTIAL = auto()  # Distance-based radial forces
    GRAVITY = auto()  # Mass-based gravitational forces
    HYBRID = auto()  # Combination of potential and gravity


class InteractionParams(TypedDict):
    """Type-safe definition of physical interaction parameters between cell types.

    All interactions use potential-based forces, while mass-based types may
    additionally use gravitational forces, creating complex hybrid behaviors.

    Attributes
    ----------
    use_potential : bool
        Whether to apply potential-based forces
    use_gravity : bool
        Whether to apply gravity-based forces
    potential_strength : float
        Positive = repulsion, Negative = attraction
    gravity_factor : float
        Strength multiplier for gravitational force
    max_dist : float
        Maximum interaction distance cutoff
    """

    use_potential: bool  # Whether to apply potential-based forces
    use_gravity: bool  # Whether to apply gravity-based forces
    potential_strength: float  # Positive = repulsion, Negative = attraction
    gravity_factor: float  # Strength multiplier for gravitational force
    max_dist: float  # Maximum interaction distance cutoff


class CellularTypeData:
    """Represents a cellular type with multiple cellular components.

    Manages positions, velocities, energy, mass, and genetic traits of components
    with vectorized operations and spatial optimization.

    Conforms to CellularTypeProtocol for rendering compatibility.

    Attributes:
        type_id: Unique identifier for this cellular type
        color: RGB color tuple for rendering components of this type
        mass_based: Whether this type uses mass in physical calculations
        x: X-coordinate positions of all components
        y: Y-coordinate positions of all components
        z: Z-coordinate positions of all components (depth, 3D only)
        vx: X-velocity components of all particles
        vy: Y-velocity components of all particles
        vz: Z-velocity components of all particles (depth, 3D only)
        energy: Energy levels of all components
        mass: Mass values for mass-based types (None for massless types)
        alive: Boolean mask indicating which components are alive
        age: Current age of each component
        max_age: Maximum age before component death
        predation_efficiency: Efficiency at extracting energy from prey
        cooldown: Recovery time after predatory actions
        base_mass: Reference mass for growth calculations
        spatial_dimensions: Dimensionality of simulation (2 or 3)
        window_depth: Depth of simulation volume for 3D mode
        plus various genetic trait arrays and metadata attributes
    """

    def __init__(
        self,
        type_id: int,
        color: ColorRGB,
        n_particles: int,
        window_width: int,
        window_height: int,
        initial_energy: float,
        max_age: float = np.inf,
        mass: Optional[float] = None,
        base_velocity_scale: float = 1.0,
        energy_efficiency: Optional[float] = None,
        gene_traits: Optional[List[str]] = None,
        gene_mutation_rate: float = 0.05,
        gene_mutation_range: Range = (-0.1, 0.1),
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
        initial_predation_efficiency: float = 0.3,
        initial_cooldown: float = 0.0,
        window_depth: Optional[int] = None,
        spatial_dimensions: int = 2,
    ) -> None:
        """Initialize a CellularTypeData instance with given parameters.

        Creates a new cellular type with specified number of components,
        initializing all trait arrays and state variables using vectorized operations.

        Args:
            type_id: Unique identifier for the cellular type
            color: RGB color tuple for rendering cellular components
            n_particles: Initial number of cellular components
            window_width: Width of the simulation window in pixels
            window_height: Height of the simulation window in pixels
            initial_energy: Initial energy assigned to each component
            max_age: Maximum age before component death
            mass: Mass of components (if mass-based)
            base_velocity_scale: Base scaling factor for initial velocities
            energy_efficiency: Initial energy efficiency trait
            gene_traits: List of gene trait names to initialize
            gene_mutation_rate: Base mutation rate for gene traits
            gene_mutation_range: Range for gene trait mutations
            min_energy: Minimum allowed energy value
            max_energy: Maximum allowed energy value
            min_mass: Minimum allowed mass value
            max_mass: Maximum allowed mass value
            min_velocity: Minimum allowed velocity value
            max_velocity: Maximum allowed velocity value
            min_perception: Minimum allowed perception range
            max_perception: Maximum allowed perception range
            min_reproduction: Minimum allowed reproduction rate
            max_reproduction: Maximum allowed reproduction rate
            min_synergy: Minimum allowed synergy affinity
            max_synergy: Maximum allowed synergy affinity
            min_colony: Minimum allowed colony factor
            max_colony: Maximum allowed colony factor
            min_drift: Maximum allowed drift sensitivity
            max_drift: Maximum allowed drift sensitivity
            min_energy_efficiency: Minimum allowed energy efficiency
            max_energy_efficiency: Maximum allowed energy efficiency
            initial_predation_efficiency: Starting predation efficiency
            initial_cooldown: Initial cooldown value for actions
            window_depth: Depth of the simulation volume for 3D mode
            spatial_dimensions: Dimensionality of the simulation (2 or 3)
        """
        # Store metadata
        self.type_id: int = type_id
        self.color: ColorRGB = color
        self.mass_based: bool = mass is not None
        self.spatial_dimensions: int = int(spatial_dimensions)
        self.window_width: int = int(window_width)
        self.window_height: int = int(window_height)
        if self.spatial_dimensions not in (2, 3):
            raise ValueError("spatial_dimensions must be 2 or 3")

        # Normalize window depth for 3D mode (defaults to max planar dimension)
        if self.spatial_dimensions == 3:
            if window_depth is None:
                window_depth = int(max(window_width, window_height))
            if window_depth <= 0:
                raise ValueError("window_depth must be positive for 3D mode")
        self.window_depth: Optional[int] = window_depth

        # Default gene traits if none provided
        if gene_traits is None:
            gene_traits = [
                "speed_factor",
                "interaction_strength",
                "perception_range",
                "reproduction_rate",
                "synergy_affinity",
                "colony_factor",
                "drift_sensitivity",
            ]

        # Store parameter bounds
        self.min_energy: float = min_energy
        self.max_energy: float = max_energy
        self.min_mass: float = min_mass
        self.max_mass: float = max_mass
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.min_perception: float = min_perception
        self.max_perception: float = max_perception
        self.min_reproduction: float = min_reproduction
        self.max_reproduction: float = max_reproduction
        self.min_synergy: float = min_synergy
        self.max_synergy: float = max_synergy
        self.min_colony: float = min_colony
        self.max_colony: float = max_colony
        self.min_drift: float = min_drift
        self.max_drift: float = max_drift
        self.min_energy_efficiency: float = min_energy_efficiency
        self.max_energy_efficiency: float = max_energy_efficiency

        # Initialize cellular component positions randomly within the window/volume
        if self.spatial_dimensions == 3:
            coords3: FloatArray = random_xyz(
                window_width, window_height, window_depth or 1, n_particles
            )
            self.x = coords3[:, 0]
            self.y = coords3[:, 1]
            self.z = coords3[:, 2]
        else:
            coords2: FloatArray = random_xy(window_width, window_height, n_particles)
            self.x = coords2[:, 0]
            self.y = coords2[:, 1]
            # Keep z for API consistency in 2D mode
            self.z = np.zeros(n_particles, dtype=np.float64)

        # Initialize energy efficiency trait
        if energy_efficiency is None:
            # Randomly initialize energy efficiency within the defined range
            self.energy_efficiency: FloatArray = np.random.uniform(
                self.min_energy_efficiency, self.max_energy_efficiency, n_particles
            ).astype(np.float64)
        else:
            # Set a fixed energy efficiency if provided
            self.energy_efficiency = np.full(
                n_particles, energy_efficiency, dtype=np.float64
            )

        # Calculate velocity scaling based on energy efficiency to prevent division by zero
        velocity_scaling: FloatArray = np.zeros(n_particles, dtype=np.float64)
        nonzero_mask = self.energy_efficiency != 0
        velocity_scaling[nonzero_mask] = (
            base_velocity_scale / self.energy_efficiency[nonzero_mask]
        )
        velocity_scaling[~nonzero_mask] = base_velocity_scale

        # Initialize cellular component velocities with random values scaled by velocity_scaling
        self.vx: FloatArray = np.clip(
            np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
            self.min_velocity,
            self.max_velocity,
        ).astype(np.float64)

        self.vy: FloatArray = np.clip(
            np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
            self.min_velocity,
            self.max_velocity,
        ).astype(np.float64)
        if self.spatial_dimensions == 3:
            self.vz: FloatArray = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity,
            ).astype(np.float64)
        else:
            self.vz = np.zeros(n_particles, dtype=np.float64)

        # Initialize energy levels for all cellular components
        self.energy: FloatArray = np.clip(
            np.full(n_particles, initial_energy, dtype=np.float64),
            self.min_energy,
            self.max_energy,
        )

        # Initialize mass if type is mass-based
        if self.mass_based:
            if mass is None or mass <= 0.0:
                raise ValueError("Mass must be positive for mass-based cellular types.")
            self.mass: Optional[FloatArray] = np.clip(
                np.full(n_particles, mass, dtype=np.float64),
                self.min_mass,
                self.max_mass,
            )
            # Store reference base mass for growth calculations
            self.base_mass: Optional[FloatArray] = self.mass.copy()
        else:
            # Explicitly set to None for non-mass-based types
            self.mass = None
            self.base_mass = None

        # Initialize alive status and age for cellular components
        self.alive: BoolArray = np.ones(n_particles, dtype=bool)
        self.age: FloatArray = np.zeros(n_particles, dtype=np.float64)
        self.max_age: float = max_age

        # Initialize gene traits
        self.speed_factor: FloatArray = np.random.uniform(0.5, 1.5, n_particles).astype(
            np.float64
        )
        self.interaction_strength: FloatArray = np.random.uniform(
            0.5, 1.5, n_particles
        ).astype(np.float64)
        self.perception_range: FloatArray = np.clip(
            np.random.uniform(50.0, 150.0, n_particles),
            self.min_perception,
            self.max_perception,
        ).astype(np.float64)
        self.reproduction_rate: FloatArray = np.clip(
            np.random.uniform(0.1, 0.5, n_particles),
            self.min_reproduction,
            self.max_reproduction,
        ).astype(np.float64)
        self.synergy_affinity: FloatArray = np.clip(
            np.random.uniform(0.5, 1.5, n_particles),
            self.min_synergy,
            self.max_synergy,
        ).astype(np.float64)
        self.colony_factor: FloatArray = np.clip(
            np.random.uniform(0.0, 1.0, n_particles),
            self.min_colony,
            self.max_colony,
        ).astype(np.float64)
        self.drift_sensitivity: FloatArray = np.clip(
            np.random.uniform(0.5, 1.5, n_particles), self.min_drift, self.max_drift
        ).astype(np.float64)

        # Gene mutation parameters
        self.gene_mutation_rate: float = gene_mutation_rate
        self.gene_mutation_range: Range = gene_mutation_range

        # Speciation and lineage tracking
        self.species_id: IntArray = np.full(n_particles, type_id, dtype=np.int_)
        self.parent_id: IntArray = np.full(n_particles, -1, dtype=np.int_)
        self.colony_id: IntArray = np.full(n_particles, -1, dtype=np.int_)
        self.colony_role: IntArray = np.zeros(n_particles, dtype=np.int_)

        # Initialize synergy connection matrix
        self.synergy_connections: BoolArray = np.zeros(
            (n_particles, n_particles), dtype=bool
        )

        # Adaptation metrics
        self.fitness_score: FloatArray = np.zeros(n_particles, dtype=np.float64)
        self.generation: IntArray = np.zeros(n_particles, dtype=np.int_)
        self.mutation_history: List[List[Tuple[str, float]]] = [
            [] for _ in range(n_particles)
        ]

        # Initialize predation mechanics
        self.predation_efficiency: FloatArray = np.full(
            n_particles, initial_predation_efficiency, dtype=np.float64
        )
        self.cooldown: FloatArray = np.full(
            n_particles, initial_cooldown, dtype=np.float64
        )

    @eidosian()
    def is_alive_mask(self) -> BoolArray:
        """Compute a mask of alive cellular components based on conditions.

        Determines which components are alive based on energy, age, and mass
        constraints, vectorizing the computation for efficiency.

        Returns:
            BoolArray: Boolean array indicating alive status of components
        """
        # Basic alive conditions: energy > 0, age < max_age, and alive flag is True
        mask: BoolArray = self.alive & (self.energy > 0.0) & (self.age < self.max_age)

        if self.mass_based and self.mass is not None:
            # Additional condition for mass-based types: mass > 0
            mask = mask & (self.mass > 0.0)

        return mask

    @eidosian()
    def update_alive(self) -> None:
        """Update the alive status of cellular components based on current conditions."""
        self.alive = self.is_alive_mask()

    @eidosian()
    def age_components(self) -> None:
        """Increment the age of each cellular component.

        Applies aging to all components and ensures energy values remain valid.
        """
        self.age += 1.0
        self.energy = np.clip(self.energy, 0.0, None)

    @eidosian()
    def update_states(self) -> None:
        """Update the state of cellular components.

        Currently a placeholder for future state management functionality.
        """
        pass

    @eidosian()
    def remove_dead(self, config: "SimulationConfig") -> None:
        """Remove dead cellular components and redistribute their energy.

        Performs energy transfer from components dying of old age to their
        nearest living neighbors, then filters all component attributes to
        keep only living components.

        Args:
            config: Simulation configuration parameters
        """
        # Validate array sizes first to prevent index mismatches
        self._synchronize_arrays()

        # Get current alive status
        alive_mask: BoolArray = self.is_alive_mask()
        if alive_mask.size > 0 and bool(np.all(alive_mask)):
            return

        # Process energy transfer for components dying of old age
        self._process_energy_transfer(alive_mask, config)

        # Apply filtering to all arrays
        self._filter_arrays(alive_mask)

    @eidosian()
    def filter_by_mask(self, mask: BoolArray) -> None:
        """Filter component arrays using a provided mask.

        Args:
            mask: Boolean array indicating which components to keep
        """
        self._synchronize_arrays()
        self._filter_arrays(mask)

    def _synchronize_arrays(self) -> None:
        """Ensure all component arrays have matching dimensions.

        Synchronizes the size of all attribute arrays to prevent index errors.
        """
        array_attributes: List[str] = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "energy",
            "alive",
            "age",
            "energy_efficiency",
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
            "species_id",
            "parent_id",
            "colony_id",
            "colony_role",
            "fitness_score",
            "generation",
            "predation_efficiency",
            "cooldown",
        ]

        # Find the smallest valid size
        sizes: List[int] = []
        for attr in array_attributes:
            if hasattr(self, attr):
                arr = getattr(self, attr)
                if arr is not None and hasattr(arr, "size"):
                    sizes.append(arr.size)

        if not sizes:
            return  # No arrays to synchronize

        min_size: int = min(sizes)

        # Trim all arrays to the smallest size
        for attr in array_attributes:
            if hasattr(self, attr):
                arr = getattr(self, attr)
                if arr is not None and hasattr(arr, "size"):
                    if arr.size > min_size:
                        setattr(self, attr, arr[:min_size])

        # Handle mass and base_mass arrays separately
        if self.mass_based:
            for mass_attr in ["mass", "base_mass"]:
                mass_array = getattr(self, mass_attr, None)
                if (
                    mass_array is not None
                    and hasattr(mass_array, "size")
                    and mass_array.size > min_size
                ):
                    setattr(self, mass_attr, mass_array[:min_size])

    def _process_energy_transfer(
        self, alive_mask: BoolArray, config: "SimulationConfig"
    ) -> None:
        """Transfer energy from dying components to nearby living components.

        Args:
            alive_mask: Boolean array indicating which components are alive
            config: Simulation configuration parameters
        """
        # Identify components dying of old age
        dead_due_to_age: BoolArray = (~alive_mask) & (self.age >= self.max_age)

        if not np.any(dead_due_to_age):
            return  # No components dying of old age

        # Find indices of alive and dead-by-age components
        alive_indices: IntArray = np.where(alive_mask)[0]
        dead_age_indices: IntArray = np.where(dead_due_to_age)[0]

        if alive_indices.size == 0:
            return  # No alive components to receive energy

        # Build positions array for KD-Tree
        if self.spatial_dimensions == 3:
            alive_positions = np.column_stack(
                (
                    self.x[alive_indices],
                    self.y[alive_indices],
                    self.z[alive_indices],
                )
            )
            world_size = (float(self.window_width), float(self.window_height), float(self.window_depth or 1))
        else:
            alive_positions = np.column_stack(
                (self.x[alive_indices], self.y[alive_indices])
            )
            world_size = (float(self.window_width), float(self.window_height))

        # Create KD-Tree with our properly typed wrapper
        tree: KDTree
        if config.boundary_mode == "wrap":
            index_map = None
            try:
                alive_positions = alive_positions.copy()
                alive_positions[:, 0] = wrap_positions(alive_positions[:, 0], 0.0, world_size[0])
                alive_positions[:, 1] = wrap_positions(alive_positions[:, 1], 0.0, world_size[1])
                if self.spatial_dimensions == 3:
                    alive_positions[:, 2] = wrap_positions(alive_positions[:, 2], 0.0, world_size[2])
                tree = KDTree(alive_positions, boxsize=world_size)
            except TypeError:
                tiled_positions, index_map = tile_positions_for_wrap(
                    alive_positions, world_size
                )
                index_map = alive_indices[index_map]
                tree = KDTree(tiled_positions)
        else:
            index_map = alive_indices
            tree = KDTree(alive_positions)

        # Process energy transfer in batches for performance
        batch_size: int = min(1000, dead_age_indices.size)
        for i in range(0, dead_age_indices.size, batch_size):
            batch_indices: IntArray = dead_age_indices[i : i + batch_size]
            if self.spatial_dimensions == 3:
                dead_positions = np.column_stack(
                    (
                        self.x[batch_indices],
                        self.y[batch_indices],
                        self.z[batch_indices],
                    )
                )
            else:
                dead_positions = np.column_stack(
                    (self.x[batch_indices], self.y[batch_indices])
                )
            dead_energies: FloatArray = self.energy[batch_indices]

            if config.boundary_mode == "wrap" and index_map is None:
                dead_positions = dead_positions.copy()
                dead_positions[:, 0] = wrap_positions(dead_positions[:, 0], 0.0, world_size[0])
                dead_positions[:, 1] = wrap_positions(dead_positions[:, 1], 0.0, world_size[1])
                if self.spatial_dimensions == 3:
                    dead_positions[:, 2] = wrap_positions(dead_positions[:, 2], 0.0, world_size[2])
                neighbor_lists = tree.query_ball_point(
                    dead_positions, config.predation_range
                )
            elif config.boundary_mode == "wrap":
                neighbor_lists = tree.query_ball_point(
                    dead_positions, config.predation_range
                )
            else:
                distances, neighbors = tree.query(
                    dead_positions,
                    k=min(3, alive_indices.size),  # Don't request more neighbors than exist
                    distance_upper_bound=config.predation_range,
                )
                neighbor_lists = [neighbors[j] for j in range(len(batch_indices))]

            # Process energy transfer for each dead component
            for j in range(len(batch_indices)):
                dead_energy = dead_energies[j]
                neighbor_indices = neighbor_lists[j]
                if len(neighbor_indices) == 0:
                    continue

                if index_map is None:
                    mapped_neighbors = np.array(neighbor_indices, dtype=np.int_)
                else:
                    mapped_neighbors = np.array(
                        [int(index_map[idx]) for idx in neighbor_indices], dtype=np.int_
                    )
                    mapped_neighbors = np.unique(mapped_neighbors)

                if mapped_neighbors.size == 0:
                    continue

                # Compute wrapped distances to filter by range
                if self.spatial_dimensions == 3:
                    delta = np.column_stack(
                        (
                            self.x[mapped_neighbors] - dead_positions[j, 0],
                            self.y[mapped_neighbors] - dead_positions[j, 1],
                            self.z[mapped_neighbors] - dead_positions[j, 2],
                        )
                    )
                    if config.boundary_mode == "wrap":
                        delta[:, 0] = wrap_deltas(delta[:, 0], world_size[0])
                        delta[:, 1] = wrap_deltas(delta[:, 1], world_size[1])
                        delta[:, 2] = wrap_deltas(delta[:, 2], world_size[2])
                else:
                    delta = np.column_stack(
                        (
                            self.x[mapped_neighbors] - dead_positions[j, 0],
                            self.y[mapped_neighbors] - dead_positions[j, 1],
                        )
                    )
                    if config.boundary_mode == "wrap":
                        delta[:, 0] = wrap_deltas(delta[:, 0], world_size[0])
                        delta[:, 1] = wrap_deltas(delta[:, 1], world_size[1])

                dist = np.sqrt(np.sum(delta * delta, axis=1))
                valid_mask = dist < config.predation_range
                if not np.any(valid_mask):
                    continue

                valid_neighbors = mapped_neighbors[valid_mask]
                energy_share: float = float(dead_energy / np.sum(valid_mask))

                for alive_idx in valid_neighbors:
                    if alive_idx < len(self.energy):
                        self.energy[alive_idx] += energy_share

                self.energy[batch_indices[j]] = 0.0

    def _filter_arrays(self, mask: BoolArray) -> None:
        """
        Filter all component arrays to keep only alive components.

        Args:
            mask: Boolean array indicating which components are alive
        """
        mask = mask.astype(bool, copy=False)
        mask_len = int(mask.shape[0])
        if mask_len == 0:
            return
        if bool(np.all(mask)):
            return
        arrays_to_filter: List[str] = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "energy",
            "alive",
            "age",
            "energy_efficiency",
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
            "species_id",
            "parent_id",
            "colony_id",
            "colony_role",
            "fitness_score",
            "generation",
            "predation_efficiency",
            "cooldown",
        ]

        # Filter each array to keep only alive components
        for attr in arrays_to_filter:
            current = getattr(self, attr, None)
            if current is None or not hasattr(current, "shape"):
                continue
            if current.shape[0] != mask_len:
                if current.shape[0] < mask_len:
                    continue
                current = current[:mask_len]
            setattr(self, attr, current[mask])

        # Handle mass and base_mass arrays separately if they exist
        if self.mass_based:
            for mass_attr in ["mass", "base_mass"]:
                mass_array = getattr(self, mass_attr, None)
                if mass_array is not None:
                    if mass_array.shape[0] != mask_len:
                        if mass_array.shape[0] < mask_len:
                            continue
                        mass_array = mass_array[:mask_len]
                    setattr(self, mass_attr, mass_array[mask])

        # Filter mutation history list
        if len(self.mutation_history) > 0:
            # Get indices as numpy array, then convert to a properly typed Python list
            indices_array: IntArray = np.where(mask)[0]
            if indices_array.size > 0:
                new_history = [
                    self.mutation_history[int(i)]
                    for i in indices_array
                    if int(i) < len(self.mutation_history)
                ]
                self.mutation_history = new_history
            else:
                self.mutation_history = []

        # Resize synergy connections matrix
        if (
            hasattr(self, "synergy_connections")
            and self.synergy_connections.shape[0] > 0
        ):
            alive_indices_array: IntArray = np.where(mask)[0]
            if alive_indices_array.size == 0:
                self.synergy_connections = np.zeros((0, 0), dtype=bool)
            else:
                max_index = self.synergy_connections.shape[0]
                alive_indices_array = alive_indices_array[alive_indices_array < max_index]
                self.synergy_connections = self.synergy_connections[
                    np.ix_(alive_indices_array, alive_indices_array)
                ]

    @eidosian()
    def add_component(
        self,
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
        predation_efficiency_val: float = 0.3,
        cooldown_val: float = 0.0,
        z: float = 0.0,
        vz: float = 0.0,
    ) -> None:
        """Add a new cellular component to this cellular type.

        Appends a new component with specified attributes to the cellular type,
        extending all relevant array attributes.

        Args:
            x: X-coordinate of the new component
            y: Y-coordinate of the new component
            vx: X-component of the new component's velocity
            vy: Y-component of the new component's velocity
            energy: Initial energy of the new component
            mass_val: Mass of the new component if type is mass-based; else None
            energy_efficiency_val: Energy efficiency trait of the new component
            speed_factor_val: Speed factor gene trait of the new component
            interaction_strength_val: Interaction strength gene trait of the new component
            perception_range_val: Perception range gene trait of the new component
            reproduction_rate_val: Reproduction rate gene trait of the new component
            synergy_affinity_val: Synergy affinity gene trait of the new component
            colony_factor_val: Colony formation gene trait of the new component
            drift_sensitivity_val: Drift sensitivity gene trait of the new component
            species_id_val: Species ID of the new component
            parent_id_val: Parent ID of the new component
            max_age: Maximum age for the new component
            predation_efficiency_val: Predation efficiency trait of the new component
            cooldown_val: Initial cooldown value for the new component
            z: Z-coordinate of the new component (depth)
            vz: Z-component of the new component's velocity
        """
        # Append new component's attributes using NumPy's concatenate for efficiency
        self.x = np.concatenate((self.x, np.array([x], dtype=np.float64)))
        self.y = np.concatenate((self.y, np.array([y], dtype=np.float64)))
        self.z = np.concatenate((self.z, np.array([z], dtype=np.float64)))
        self.vx = np.concatenate((self.vx, np.array([vx], dtype=np.float64)))
        self.vy = np.concatenate((self.vy, np.array([vy], dtype=np.float64)))
        self.vz = np.concatenate((self.vz, np.array([vz], dtype=np.float64)))
        self.energy = np.concatenate(
            (self.energy, np.array([energy], dtype=np.float64))
        )
        self.alive = np.concatenate((self.alive, np.array([True], dtype=bool)))
        self.age = np.concatenate((self.age, np.array([0.0], dtype=np.float64)))
        self.energy_efficiency = np.concatenate(
            (
                self.energy_efficiency,
                np.array([energy_efficiency_val], dtype=np.float64),
            )
        )
        self.speed_factor = np.concatenate(
            (self.speed_factor, np.array([speed_factor_val], dtype=np.float64))
        )
        self.interaction_strength = np.concatenate(
            (
                self.interaction_strength,
                np.array([interaction_strength_val], dtype=np.float64),
            )
        )
        self.perception_range = np.concatenate(
            (self.perception_range, np.array([perception_range_val], dtype=np.float64))
        )
        self.reproduction_rate = np.concatenate(
            (
                self.reproduction_rate,
                np.array([reproduction_rate_val], dtype=np.float64),
            )
        )
        self.synergy_affinity = np.concatenate(
            (self.synergy_affinity, np.array([synergy_affinity_val], dtype=np.float64))
        )
        self.colony_factor = np.concatenate(
            (self.colony_factor, np.array([colony_factor_val], dtype=np.float64))
        )
        self.drift_sensitivity = np.concatenate(
            (
                self.drift_sensitivity,
                np.array([drift_sensitivity_val], dtype=np.float64),
            )
        )
        self.species_id = np.concatenate(
            (self.species_id, np.array([species_id_val], dtype=np.int_))
        )
        self.parent_id = np.concatenate(
            (self.parent_id, np.array([parent_id_val], dtype=np.int_))
        )

        # Append predation traits
        self.predation_efficiency = np.concatenate(
            (
                self.predation_efficiency,
                np.array([predation_efficiency_val], dtype=np.float64),
            )
        )
        self.cooldown = np.concatenate(
            (self.cooldown, np.array([cooldown_val], dtype=np.float64))
        )

        # Handle mass for mass-based types
        if self.mass_based and self.mass is not None:
            if mass_val is None or mass_val <= 0.0:
                # Ensure mass is positive; assign a small random mass if invalid
                mass_val = max(0.1, abs(mass_val if mass_val is not None else 1.0))
            self.mass = np.concatenate(
                (self.mass, np.array([mass_val], dtype=np.float64))
            )

            # Update base_mass as well if it exists
            if hasattr(self, "base_mass") and self.base_mass is not None:
                self.base_mass = np.concatenate(
                    (self.base_mass, np.array([mass_val], dtype=np.float64))
                )

        # Update mutation history list
        self.mutation_history.append([])

        # Update synergy connections matrix
        old_size: int = self.synergy_connections.shape[0]
        new_size: int = old_size + 1
        new_connections: BoolArray = np.zeros((new_size, new_size), dtype=bool)
        new_connections[:old_size, :old_size] = self.synergy_connections
        self.synergy_connections = new_connections

        # Update colony role
        self.colony_role = np.concatenate(
            (self.colony_role, np.array([0], dtype=np.int_))
        )

        # Update colony ID
        self.colony_id = np.concatenate((self.colony_id, np.array([-1], dtype=np.int_)))

        # Update fitness score
        self.fitness_score = np.concatenate(
            (self.fitness_score, np.array([0.0], dtype=np.float64))
        )

        # Update generation
        parent_gen: int = 0
        if parent_id_val >= 0 and parent_id_val < len(self.generation):
            parent_gen = int(self.generation[parent_id_val])  # Ensure this is an int
        self.generation = np.concatenate(
            (self.generation, np.array([parent_gen + 1], dtype=np.int_))
        )

    @eidosian()
    def add_components_bulk(
        self,
        x: FloatArray,
        y: FloatArray,
        vx: FloatArray,
        vy: FloatArray,
        energy: FloatArray,
        mass: Optional[FloatArray],
        energy_efficiency: FloatArray,
        speed_factor: FloatArray,
        interaction_strength: FloatArray,
        perception_range: FloatArray,
        reproduction_rate: FloatArray,
        synergy_affinity: FloatArray,
        colony_factor: FloatArray,
        drift_sensitivity: FloatArray,
        species_id: IntArray,
        parent_id: IntArray,
        predation_efficiency: Optional[FloatArray] = None,
        cooldown: Optional[FloatArray] = None,
        z: Optional[FloatArray] = None,
        vz: Optional[FloatArray] = None,
    ) -> None:
        """Add multiple new components in a single vectorized operation."""
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        vx_arr = np.asarray(vx, dtype=np.float64).reshape(-1)
        vy_arr = np.asarray(vy, dtype=np.float64).reshape(-1)
        if z is None:
            z_arr = np.zeros_like(x_arr, dtype=np.float64)
        else:
            z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
        if vz is None:
            vz_arr = np.zeros_like(x_arr, dtype=np.float64)
        else:
            vz_arr = np.asarray(vz, dtype=np.float64).reshape(-1)
        energy_arr = np.asarray(energy, dtype=np.float64).reshape(-1)
        energy_eff_arr = np.asarray(energy_efficiency, dtype=np.float64).reshape(-1)
        speed_arr = np.asarray(speed_factor, dtype=np.float64).reshape(-1)
        interaction_arr = np.asarray(interaction_strength, dtype=np.float64).reshape(-1)
        perception_arr = np.asarray(perception_range, dtype=np.float64).reshape(-1)
        reproduction_arr = np.asarray(reproduction_rate, dtype=np.float64).reshape(-1)
        synergy_arr = np.asarray(synergy_affinity, dtype=np.float64).reshape(-1)
        colony_arr = np.asarray(colony_factor, dtype=np.float64).reshape(-1)
        drift_arr = np.asarray(drift_sensitivity, dtype=np.float64).reshape(-1)
        species_arr = np.asarray(species_id, dtype=np.int_).reshape(-1)
        parent_arr = np.asarray(parent_id, dtype=np.int_).reshape(-1)

        count = int(x_arr.size)
        if count == 0:
            return

        expected = [
            y_arr,
            vx_arr,
            vy_arr,
            z_arr,
            vz_arr,
            energy_arr,
            energy_eff_arr,
            speed_arr,
            interaction_arr,
            perception_arr,
            reproduction_arr,
            synergy_arr,
            colony_arr,
            drift_arr,
            species_arr,
            parent_arr,
        ]
        if any(arr.size != count for arr in expected):
            raise ValueError("All component arrays must have identical lengths")

        # Append core arrays
        self.x = np.concatenate((self.x, x_arr))
        self.y = np.concatenate((self.y, y_arr))
        self.z = np.concatenate((self.z, z_arr))
        self.vx = np.concatenate((self.vx, vx_arr))
        self.vy = np.concatenate((self.vy, vy_arr))
        self.vz = np.concatenate((self.vz, vz_arr))
        self.energy = np.concatenate((self.energy, energy_arr))
        self.alive = np.concatenate((self.alive, np.ones(count, dtype=bool)))
        self.age = np.concatenate((self.age, np.zeros(count, dtype=np.float64)))
        self.energy_efficiency = np.concatenate((self.energy_efficiency, energy_eff_arr))
        self.speed_factor = np.concatenate((self.speed_factor, speed_arr))
        self.interaction_strength = np.concatenate(
            (self.interaction_strength, interaction_arr)
        )
        self.perception_range = np.concatenate((self.perception_range, perception_arr))
        self.reproduction_rate = np.concatenate((self.reproduction_rate, reproduction_arr))
        self.synergy_affinity = np.concatenate((self.synergy_affinity, synergy_arr))
        self.colony_factor = np.concatenate((self.colony_factor, colony_arr))
        self.drift_sensitivity = np.concatenate((self.drift_sensitivity, drift_arr))
        self.species_id = np.concatenate((self.species_id, species_arr))
        self.parent_id = np.concatenate((self.parent_id, parent_arr))

        # Predation traits
        if predation_efficiency is None:
            predation_efficiency = np.full(count, 0.3, dtype=np.float64)
        if cooldown is None:
            cooldown = np.zeros(count, dtype=np.float64)
        self.predation_efficiency = np.concatenate(
            (self.predation_efficiency, np.asarray(predation_efficiency, dtype=np.float64))
        )
        self.cooldown = np.concatenate(
            (self.cooldown, np.asarray(cooldown, dtype=np.float64))
        )

        # Mass handling
        if self.mass_based:
            if mass is None:
                mass_arr = np.full(count, 0.1, dtype=np.float64)
            else:
                mass_arr = np.asarray(mass, dtype=np.float64).reshape(-1)
                if mass_arr.size != count:
                    raise ValueError("Mass array length must match component count")
                mass_arr = np.where(mass_arr > 0.0, mass_arr, 0.1)
            if self.mass is None:
                self.mass = mass_arr
            else:
                self.mass = np.concatenate((self.mass, mass_arr))
            if hasattr(self, "base_mass") and self.base_mass is not None:
                self.base_mass = np.concatenate((self.base_mass, mass_arr))
        else:
            self.mass = None

        # Mutation history and synergy connections
        self.mutation_history.extend([[] for _ in range(count)])
        old_size = self.synergy_connections.shape[0]
        new_size = old_size + count
        new_connections: BoolArray = np.zeros((new_size, new_size), dtype=bool)
        if old_size > 0:
            new_connections[:old_size, :old_size] = self.synergy_connections
        self.synergy_connections = new_connections

        # Colony and fitness metadata
        self.colony_role = np.concatenate((self.colony_role, np.zeros(count, dtype=np.int_)))
        self.colony_id = np.concatenate((self.colony_id, np.full(count, -1, dtype=np.int_)))
        self.fitness_score = np.concatenate((self.fitness_score, np.zeros(count, dtype=np.float64)))

        # Generation tracking
        parent_gen = np.zeros(count, dtype=np.int_)
        valid_parents = (parent_arr >= 0) & (parent_arr < self.generation.size)
        if np.any(valid_parents):
            parent_gen[valid_parents] = self.generation[parent_arr[valid_parents]]
        self.generation = np.concatenate((self.generation, parent_gen + 1))
