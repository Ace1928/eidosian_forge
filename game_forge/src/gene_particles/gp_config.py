"""
GeneParticles: Cellular Automata with Dynamic Gene Expression and Emergent Behaviors
------------------------------------------------------------------------------------
config.py
Simulation Config and Genetic Params Config and Simulation Constants
"""

from __future__ import annotations  # Enable self-referential type hints

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Final, List, Mapping, Tuple, cast

import numpy as np

from game_forge.src.gene_particles.gp_types import (
from eidosian_core import eidosian
    ColorRGB,
    FloatArray,
    Range,
    TraitDefinition,
    TraitType,
)

# Constants
###############################################################
# Visual Constants - Color and Layout Configuration
###############################################################

# Text colors for distinct statistic categories
FPS_COLOR: Final[ColorRGB] = (50, 255, 50)  # Bright green for performance metrics
SPECIES_COLOR: Final[ColorRGB] = (255, 180, 0)  # Orange for taxonomic information
PARTICLES_COLOR: Final[ColorRGB] = (100, 200, 255)  # Light blue for population data

# Background configuration for statistics display
STATS_BG_ALPHA: Final[int] = 120  # Semi-transparency level (0-255)
STATS_HEIGHT: Final[int] = 60  # Vertical space for statistics panel

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Physical Constants: System parameters and evolutionary constraints        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Relationship formation probabilities
SYNERGY_PROBABILITY: Final[float] = 0.1  # Baseline chance for synergy relation
GIVE_TAKE_PROBABILITY: Final[float] = 0.1  # Baseline chance for predation relation

# Evolution parameters
EVOLUTION_ADJUSTMENT_RANGE: Final[Tuple[float, float]] = (0.95, 1.05)  # ±5% mutation

# Spatial constraints
INTERACTION_DISTANCE_RANGE: Final[Tuple[float, float]] = (50.0, 200.0)  # Units
MIN_INTERACTION_DISTANCE: Final[float] = 10.0  # Minimum distance

# Force parameters
SYNERGY_STRENGTH_RANGE: Final[Tuple[float, float]] = (0.1, 0.9)  # Dimensionless factor
GRAVITY_STRENGTH_RANGE: Final[Tuple[float, float]] = (0.1, 2.0)  # Dimensionless factor


@dataclass
class GeneticParamConfig:
    """
    Configuration for genetic parameters and mutation dynamics.

    Defines trait specifications, mutation rates, and constraint mechanisms
    to maintain biologically plausible simulation behaviors. Serves as the
    central registry for all genetic trait definitions and their mutation
    parameters.

    Attributes:
        gene_traits: Registry of all genetic trait names
        gene_mutation_rate: Probability of trait mutation during reproduction (0.0-1.0)
        gene_mutation_range: Bounds for mutation magnitude as (min_delta, max_delta)
        trait_definitions: Complete specifications for all genetic traits
        energy_efficiency_mutation_rate: Probability of energy efficiency mutation (0.0-1.0)
        energy_efficiency_mutation_range: Bounds for energy efficiency mutation as (min_delta, max_delta)
    """

    gene_traits: List[str] = field(default_factory=list)
    gene_mutation_rate: float = 0.1
    gene_mutation_range: Range = (-0.05, 0.05)
    trait_definitions: Dict[str, TraitDefinition] = field(default_factory=dict)
    energy_efficiency_mutation_rate: float = 0.1
    energy_efficiency_mutation_range: Range = (-0.025, 0.05)

    # Reserved trait names that must be present in all valid configurations
    CORE_TRAITS: ClassVar[Final[List[str]]] = [
        "speed_factor",
        "interaction_strength",
        "perception_range",
        "reproduction_rate",
        "synergy_affinity",
        "colony_factor",
        "drift_sensitivity",
    ]

    def __post_init__(self) -> None:
        """Initialize trait definitions and validate configuration integrity."""
        # Define core trait specifications with full metadata
        self._initialize_trait_definitions()

        # Extract trait names from definitions for convenient access
        self.gene_traits = list(self.trait_definitions.keys())

        # Verify configuration validity
        self._validate()

    def _initialize_trait_definitions(self) -> None:
        """
        Build trait specification registry with full trait metadata.

        Each trait receives a complete definition including valid ranges,
        behavioral category, and descriptive explanation. This creates the
        genetic foundation for the simulation ecosystem.
        """
        self.trait_definitions = {
            "speed_factor": TraitDefinition(
                name="speed_factor",
                type=TraitType.MOVEMENT,
                range=(0.05, 2.0),
                description="Movement velocity multiplier",
                default=1.0,
            ),
            "interaction_strength": TraitDefinition(
                name="interaction_strength",
                type=TraitType.INTERACTION,
                range=(0.05, 2.0),
                description="Force interaction multiplier",
                default=1.0,
            ),
            "perception_range": TraitDefinition(
                name="perception_range",
                type=TraitType.PERCEPTION,
                range=(20.0, 400.0),
                description="Environmental sensing distance",
                default=200.0,
            ),
            "reproduction_rate": TraitDefinition(
                name="reproduction_rate",
                type=TraitType.REPRODUCTION,
                range=(0.02, 1.5),
                description="Probability of reproduction event",
                default=0.5,
            ),
            "synergy_affinity": TraitDefinition(
                name="synergy_affinity",
                type=TraitType.SOCIAL,
                range=(0.0, 3.0),
                description="Cooperative energy sharing strength",
                default=1.5,
            ),
            "colony_factor": TraitDefinition(
                name="colony_factor",
                type=TraitType.SOCIAL,
                range=(0.0, 2.0),
                description="Tendency to form collective groups",
                default=0.5,
            ),
            "drift_sensitivity": TraitDefinition(
                name="drift_sensitivity",
                type=TraitType.ADAPTATION,
                range=(0.0, 3.0),
                description="Responsiveness to evolutionary pressure",
                default=1.0,
            ),
        }

    def _validate(self) -> None:
        """
        Verify configuration integrity and parameter constraints.

        Performs comprehensive validation to ensure genetic parameters are
        within acceptable ranges and follow biologically plausible rules.

        Raises:
            ValueError: If any parameter violates defined constraints
        """
        # Validate mutation rates (must be probabilities between 0-1)
        self._validate_probability("gene_mutation_rate", self.gene_mutation_rate)
        self._validate_probability(
            "energy_efficiency_mutation_rate", self.energy_efficiency_mutation_rate
        )

        # Validate mutation ranges (min must be less than max)
        self._validate_range_order("gene_mutation_range", self.gene_mutation_range)
        self._validate_range_order(
            "energy_efficiency_mutation_range", self.energy_efficiency_mutation_range
        )

        # Verify all required traits are defined
        for trait in self.gene_traits:
            if trait not in self.trait_definitions:
                raise ValueError(f"Missing trait definition for '{trait}'")

    def _validate_probability(self, name: str, value: float) -> None:
        """
        Validate that a parameter represents a valid probability (0.0-1.0).

        Args:
            name: Parameter name for error reporting
            value: Value to validate

        Raises:
            ValueError: If value is outside valid probability range
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")

    def _validate_range_order(self, name: str, value: Range) -> None:
        """
        Validate that a range tuple has min < max.

        Args:
            name: Parameter name for error reporting
            value: Range tuple to validate

        Raises:
            ValueError: If min >= max
        """
        if value[0] >= value[1]:
            raise ValueError(f"{name} must have min < max, got {value}")

    @eidosian()
    def clamp_gene_values(
        self,
        speed_factor: FloatArray,
        interaction_strength: FloatArray,
        perception_range: FloatArray,
        reproduction_rate: FloatArray,
        synergy_affinity: FloatArray,
        colony_factor: FloatArray,
        drift_sensitivity: FloatArray,
    ) -> Tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """
        Constrain genetic trait arrays to their valid ranges.

        Prevents simulation instability by enforcing trait boundaries, acting as
        a genetic "immune system" against extreme mutations.

        Args:
            speed_factor: Movement velocity multipliers
            interaction_strength: Interaction force multipliers
            perception_range: Environmental sensing distances
            reproduction_rate: Reproduction probability values
            synergy_affinity: Energy sharing tendencies
            colony_factor: Group formation tendencies
            drift_sensitivity: Evolutionary responsiveness values

        Returns:
            Tuple of constrained arrays with identical shapes but values
            clamped to their respective trait definition ranges
        """
        # Apply constraint boundaries to each trait array using vectorized operations
        return (
            np.clip(
                speed_factor,
                self.trait_definitions["speed_factor"].range[0],
                self.trait_definitions["speed_factor"].range[1],
            ),
            np.clip(
                interaction_strength,
                self.trait_definitions["interaction_strength"].range[0],
                self.trait_definitions["interaction_strength"].range[1],
            ),
            np.clip(
                perception_range,
                self.trait_definitions["perception_range"].range[0],
                self.trait_definitions["perception_range"].range[1],
            ),
            np.clip(
                reproduction_rate,
                self.trait_definitions["reproduction_rate"].range[0],
                self.trait_definitions["reproduction_rate"].range[1],
            ),
            np.clip(
                synergy_affinity,
                self.trait_definitions["synergy_affinity"].range[0],
                self.trait_definitions["synergy_affinity"].range[1],
            ),
            np.clip(
                colony_factor,
                self.trait_definitions["colony_factor"].range[0],
                self.trait_definitions["colony_factor"].range[1],
            ),
            np.clip(
                drift_sensitivity,
                self.trait_definitions["drift_sensitivity"].range[0],
                self.trait_definitions["drift_sensitivity"].range[1],
            ),
        )

    @eidosian()
    def get_range_for_trait(self, trait_name: str) -> Range:
        """
        Retrieve the valid range for a specific trait.

        Args:
            trait_name: Name of the genetic trait

        Returns:
            Tuple containing (min, max) valid values

        Raises:
            KeyError: If trait_name is not a valid genetic trait
        """
        if trait_name not in self.trait_definitions:
            raise KeyError(f"Unknown trait: {trait_name}")
        return self.trait_definitions[trait_name].range

    @eidosian()
    def to_dict(self) -> Dict[str, object]:
        """
        Convert configuration to a serializable dictionary.

        Creates a representation suitable for JSON serialization, configuration
        persistence, or transmission between systems.

        Returns:
            Dictionary containing all configuration parameters with
            trait definitions converted to serializable format
        """
        # Extract all fields except trait_definitions
        result = {k: v for k, v in asdict(self).items() if k != "trait_definitions"}

        # Convert TraitDefinitions to dictionaries
        result["trait_definitions"] = {
            name: asdict(definition)
            for name, definition in self.trait_definitions.items()
        }

        return cast(Dict[str, object], result)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> GeneticParamConfig:
        """
        Create a genetic parameter configuration from a dictionary.

        Factory method that reconstructs a fully validated GeneticParamConfig
        from a previously serialized dictionary representation.

        Args:
            config_dict: Dictionary containing genetic configuration parameters

        Returns:
            Fully initialized GeneticParamConfig instance

        Raises:
            ValueError: If dictionary contains invalid genetic configuration
            TypeError: If dictionary structure doesn't match expected schema
        """
        instance = cls()
        config_copy = dict(config_dict)  # Create mutable copy

        # Handle trait definitions separately if present
        trait_defs = config_copy.pop("trait_definitions", {})
        if trait_defs:
            # Recreate TraitDefinition objects from dictionary representation
            instance.trait_definitions = {}
            for name, def_dict in cast(
                Dict[str, Dict[str, object]], trait_defs
            ).items():
                trait_type = def_dict["type"]
                if isinstance(trait_type, Enum):
                    # Handle already-enum values
                    enum_type = TraitType[trait_type.name]
                else:
                    # Handle string representation
                    enum_type = TraitType[cast(str, trait_type)]

                instance.trait_definitions[name] = TraitDefinition(
                    name=cast(str, def_dict["name"]),
                    type=enum_type,
                    range=cast(Range, def_dict["range"]),
                    description=cast(str, def_dict["description"]),
                    default=cast(float, def_dict["default"]),
                )

        # Set all other attributes
        for key, value in config_copy.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        # Perform validation to ensure reconstructed config is valid
        instance._validate()
        return instance


class SimulationConfig:
    """
    Master configuration for the GeneParticles simulation ecosystem.

    Controls all simulation parameters including population dynamics,
    physical properties, energy systems, evolutionary mechanics,
    and emergent behavior tuning.

    Attributes:
        # Population parameters
        n_cell_types: Number of distinct cellular types (int > 0)
        particles_per_type: Initial particles per cellular type (int > 0)
        min_particles_per_type: Minimum particles threshold per type (int > 0)
        max_particles_per_type: Maximum particles allowed per type (int > min)

        # Physical parameters
        mass_range: Min/max particle mass values as (min: float > 0, max: float)
        base_velocity_scale: Base movement speed multiplier (float > 0)
        mass_based_fraction: Proportion of types using mass-based interactions (0.0-1.0)
        interaction_strength_range: Min/max force strength as (min: float, max: float)
        friction: Environmental movement resistance coefficient (0.0-1.0)
        global_temperature: Random motion intensity factor (float ≥ 0)

        # Energy system parameters
        initial_energy: Starting energy per particle (float > 0)
        predation_range: Maximum distance for predation interactions (float > 0)
        energy_transfer_factor: Efficiency of energy transfers (0.0-1.0)
        mass_transfer: Whether mass transfers with energy (bool)
        max_age: Maximum particle lifespan in frames (np.inf = immortal)
        energy_efficiency_range: Min/max energy efficiency values (min: float, max: float)
        max_energy: Maximum energy capacity per particle (float > 0)

        # Lifecycle parameters
        max_frames: Maximum simulation frames (0 = infinite)
        evolution_interval: Frames between evolutionary updates (int > 0)
        synergy_range: Maximum distance for synergy interactions (float > 0)

        # Selection parameters
        culling_fitness_weights: Weighting factors for particle fitness (Dict[str, float])

        # Reproduction parameters
        reproduction_energy_threshold: Minimum energy for reproduction (float > 0)
        reproduction_mutation_rate: Mutation probability during reproduction (0.0-1.0)
        reproduction_offspring_energy_fraction: Energy fraction given to offspring (0.0-1.0)

        # Flocking behavior parameters
        alignment_strength: Strength of flocking alignment behavior (float)
        cohesion_strength: Strength of flocking cohesion behavior (float)
        separation_strength: Strength of flocking separation behavior (float)
        cluster_radius: Radius for flocking neighbor detection (float > 0)

        # Visualization parameters
        particle_size: Visual size of particles (float > 0)

        # Emergence parameters
        speciation_threshold: Genetic distance threshold for new species (float > 0)
        colony_formation_probability: Probability of colony formation (0.0-1.0)
        colony_radius: Maximum colony size (float > 0)
        colony_cohesion_strength: Force pulling colony members together (0.0-1.0)
        synergy_evolution_rate: Rate of synergy relationship changes (0.0-1.0)
        complexity_factor: Multiplier for behavior complexity (float > 0)
        structural_complexity_weight: Emphasis on emergent structures (0.0-1.0)

        # Genetic parameters
        genetics: Nested genetic parameter configuration (GeneticParamConfig)
    """

    def __init__(self) -> None:
        """
        Initialize simulation configuration with default parameters.

        Creates a fully functional configuration with balanced parameters
        suitable for most simulation scenarios. All parameters undergo
        validation to ensure simulation stability.
        """
        # Population parameters
        self.n_cell_types: int = 5
        self.particles_per_type: int = 100
        self.min_particles_per_type: int = 50
        self.max_particles_per_type: int = 1000

        # Physical parameters
        self.mass_range: Range = (1.0, 100.0)
        self.base_velocity_scale: float = 0.1
        self.mass_based_fraction: float = 1.0
        self.interaction_strength_range: Range = (-0.5, 2.0)
        self.friction: float = 0.05
        self.global_temperature: float = 0.25

        # Energy system parameters
        self.initial_energy: float = 150.0
        self.predation_range: float = 100
        self.energy_transfer_factor: float = 1.0
        self.mass_transfer: bool = True
        self.energy_efficiency_range: Range = (-0.4, 2.0)
        self.max_energy: float = 500.0

        # Lifecycle parameters
        self.max_frames: int = 0  # 0 = infinite
        self.max_age: float = np.inf
        self.evolution_interval: int = 100
        self.synergy_range: float = 100.0

        # Reproduction parameters
        self.reproduction_energy_threshold: float = 100.0
        self.reproduction_mutation_rate: float = 0.5
        self.reproduction_offspring_energy_fraction: float = 0.75

        # Flocking behavior parameters
        self.alignment_strength: float = 0.25
        self.cohesion_strength: float = 0.25
        self.separation_strength: float = 0.25
        self.cluster_radius: float = 10.0

        # Visualization parameters
        self.particle_size: float = 2.0

        # Fitness and selection parameters
        self.culling_fitness_weights: Dict[str, float] = {
            "energy_weight": 0.6,
            "age_weight": 0.8,
            "speed_factor_weight": 0.7,
            "interaction_strength_weight": 0.7,
            "synergy_affinity_weight": 0.8,
            "colony_factor_weight": 0.9,
            "drift_sensitivity_weight": 0.6,
        }

        # Emergence parameters
        self.speciation_threshold: float = 0.8
        self.colony_formation_probability: float = 0.4
        self.colony_radius: float = 50.0
        self.colony_cohesion_strength: float = 1.0
        self.synergy_evolution_rate: float = 0.08
        self.complexity_factor: float = 2.0
        self.structural_complexity_weight: float = 0.9

        # Initialize genetic parameters
        self.genetics: GeneticParamConfig = GeneticParamConfig()

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """
        Verify all configuration parameters for simulation stability.

        Performs comprehensive validation to ensure parameters are within
        acceptable ranges, preventing crashes or unexpected behaviors.

        Raises:
            ValueError: If any parameter violates its constraints
        """
        self._validate_population_parameters()
        self._validate_physical_parameters()
        self._validate_energy_parameters()
        self._validate_lifecycle_parameters()
        self._validate_reproduction_parameters()
        self._validate_flocking_parameters()
        self._validate_visualization_parameters()
        self._validate_emergence_parameters()

    def _validate_population_parameters(self) -> None:
        """Validate population-related parameters."""
        if self.n_cell_types <= 0:
            raise ValueError("Number of cell types must be greater than 0")

        if self.particles_per_type <= 0:
            raise ValueError("Particles per type must be greater than 0")

        if self.min_particles_per_type <= 0:
            raise ValueError("Minimum particles per type must be greater than 0")

        if self.max_particles_per_type < self.min_particles_per_type:
            raise ValueError("Maximum particles must be ≥ minimum particles")

    def _validate_physical_parameters(self) -> None:
        """Validate physical property parameters."""
        if self.mass_range[0] <= 0:
            raise ValueError("Minimum mass must be positive")

        if self.mass_range[0] >= self.mass_range[1]:
            raise ValueError("Mass range must have min < max")

        if self.base_velocity_scale <= 0:
            raise ValueError("Base velocity scale must be positive")

        if not (0.0 <= self.mass_based_fraction <= 1.0):
            raise ValueError("Mass-based fraction must be between 0.0 and 1.0")

        if self.interaction_strength_range[0] >= self.interaction_strength_range[1]:
            raise ValueError("Interaction strength range must have min < max")

    def _validate_energy_parameters(self) -> None:
        """Validate energy system parameters."""
        if self.initial_energy <= 0:
            raise ValueError("Initial energy must be positive")

        if not (0.0 <= self.friction <= 1.0):
            raise ValueError("Friction must be between 0.0 and 1.0")

        if self.global_temperature < 0:
            raise ValueError("Global temperature must be non-negative")

        if self.predation_range <= 0:
            raise ValueError("Predation range must be positive")

        if not (0.0 <= self.energy_transfer_factor <= 1.0):
            raise ValueError("Energy transfer factor must be between 0.0 and 1.0")

        if self.energy_efficiency_range[0] >= self.energy_efficiency_range[1]:
            raise ValueError("Energy efficiency range must have min < max")

        if self.max_energy <= 0:
            raise ValueError("Maximum energy must be positive")

    def _validate_lifecycle_parameters(self) -> None:
        """Validate lifecycle and evolution parameters."""
        if self.max_frames < 0:
            raise ValueError("Maximum frames must be non-negative")

        if self.synergy_range <= 0:
            raise ValueError("Synergy range must be positive")

        if self.evolution_interval <= 0:
            raise ValueError("Evolution interval must be positive")

    def _validate_reproduction_parameters(self) -> None:
        """Validate reproduction and mutation parameters."""
        if self.reproduction_energy_threshold <= 0:
            raise ValueError("Reproduction energy threshold must be positive")

        if not (0.0 <= self.reproduction_mutation_rate <= 1.0):
            raise ValueError("Reproduction mutation rate must be between 0.0 and 1.0")

        if not (0.0 <= self.reproduction_offspring_energy_fraction <= 1.0):
            raise ValueError(
                "Reproduction offspring energy fraction must be between 0.0 and 1.0"
            )

    def _validate_flocking_parameters(self) -> None:
        """Validate flocking and group behavior parameters."""
        if self.cluster_radius <= 0:
            raise ValueError("Cluster radius must be positive")

    def _validate_visualization_parameters(self) -> None:
        """Validate visualization and rendering parameters."""
        if self.particle_size <= 0:
            raise ValueError("Particle size must be positive")

    def _validate_emergence_parameters(self) -> None:
        """Validate complex emergence behavior parameters."""
        if self.speciation_threshold <= 0:
            raise ValueError("Speciation threshold must be positive")

        if not (0.0 <= self.colony_formation_probability <= 1.0):
            raise ValueError("Colony formation probability must be between 0.0 and 1.0")

        if self.colony_radius <= 0:
            raise ValueError("Colony radius must be positive")

        if not (0.0 <= self.colony_cohesion_strength <= 1.0):
            raise ValueError("Colony cohesion strength must be between 0.0 and 1.0")

        if not (0.0 <= self.synergy_evolution_rate <= 1.0):
            raise ValueError("Synergy evolution rate must be between 0.0 and 1.0")

        if self.complexity_factor <= 0:
            raise ValueError("Complexity factor must be positive")

        if not (0.0 <= self.structural_complexity_weight <= 1.0):
            raise ValueError("Structural complexity weight must be between 0.0 and 1.0")

    @eidosian()
    def to_dict(self) -> Dict[str, object]:
        """
        Convert entire configuration to a flat dictionary representation.

        Creates a serializable representation of all parameters including
        nested configurations for logging, persistence, or visualization.

        Returns:
            Dictionary containing all configuration parameters with
            nested genetics configuration as nested dictionary
        """
        # Start with all attributes except genetics
        config_dict = {k: v for k, v in self.__dict__.items() if k != "genetics"}

        # Add genetics as a nested dictionary
        config_dict["genetics"] = self.genetics.to_dict()

        return cast(Dict[str, object], config_dict)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> SimulationConfig:
        """
        Create a configuration instance from a dictionary.

        Factory method that reconstructs a fully validated SimulationConfig
        from a previously serialized dictionary representation.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Fully initialized SimulationConfig instance

        Raises:
            ValueError: If dictionary contains invalid configuration
            TypeError: If dictionary structure doesn't match expected schema
        """
        instance = cls()
        config_copy = dict(config_dict)  # Create mutable copy

        # Extract genetics config if present and properly initialize it
        genetics_dict = config_copy.pop("genetics", None)
        if genetics_dict:
            instance.genetics = GeneticParamConfig.from_dict(
                cast(Mapping[str, object], genetics_dict)
            )

        # Set all other attributes
        for key, value in config_copy.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        # Validate after setting all values
        instance._validate()
        return instance
