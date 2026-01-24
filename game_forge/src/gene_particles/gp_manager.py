"""Gene Particles Evolution Management System.

Provides core evolutionary mechanics for the Gene Particles simulation,
including mutation, reproduction, selection, speciation, and adaptation
with rigorous type safety and vectorized operations.
"""

from typing import Dict, List, Optional

import numpy as np

from game_forge.src.gene_particles.gp_config import SimulationConfig
from game_forge.src.gene_particles.gp_types import (
from eidosian_core import eidosian
    BoolArray,
    CellularTypeData,
    ColorRGB,
    FloatArray,
    IntArray,
)
from game_forge.src.gene_particles.gp_utility import mutate_trait

###############################################################
# Cellular Type Manager (Handles Multi-Type Operations & Reproduction)
###############################################################


class CellularTypeManager:
    """
    Manages all cellular types in the simulation ecosystem.

    Centralizes operations that span multiple cellular types, including
    reproduction with trait inheritance, mutation, speciation, and population
    control. Maintains type relationships and handles genetic evolution.

    Attributes:
        config: Simulation configuration parameters
        cellular_types: List of all cellular type data instances
        mass_based_type_indices: Indices of types that use mass-based physics
        colors: RGB color tuples for visual representation of each type
    """

    def __init__(
        self,
        config: SimulationConfig,
        colors: List[ColorRGB],
        mass_based_type_indices: List[int],
    ) -> None:
        """
        Initialize the CellularTypeManager with configuration and type settings.

        Args:
            config: Simulation configuration parameters
            colors: List of RGB color tuples for each cellular type
            mass_based_type_indices: Indices of types that use mass-based physics
        """
        self.config: SimulationConfig = config
        self.cellular_types: List[CellularTypeData] = []
        self.mass_based_type_indices: List[int] = mass_based_type_indices
        self.colors: List[ColorRGB] = colors

    @eidosian()
    def add_cellular_type_data(self, data: CellularTypeData) -> None:
        """
        Add a cellular type to the manager.

        Args:
            data: The cellular type data instance to add
        """
        self.cellular_types.append(data)

    @eidosian()
    def get_cellular_type_by_id(self, i: int) -> CellularTypeData:
        """
        Retrieve a cellular type by its index.

        Args:
            i: Index of the cellular type to retrieve

        Returns:
            The corresponding cellular type data instance
        """
        return self.cellular_types[i]

    @eidosian()
    def remove_dead_in_all_types(self) -> None:
        """
        Remove dead cellular components from all managed types.

        Iterates through all cellular types and removes components that
        no longer meet the survival criteria (energy > 0, alive flag, etc.).
        """
        for ct in self.cellular_types:
            ct.remove_dead(self.config)

    @eidosian()
    def reproduce(self) -> None:
        """
        Handle reproduction across all cellular types.

        Implements the core genetic algorithm of the simulation, including:
        - Identifying components eligible for reproduction
        - Energy transfer from parent to offspring
        - Trait inheritance with mutations
        - Speciation based on genetic distance
        - Population management

        All operations are vectorized for performance where possible.
        """
        # Process each cellular type independently
        for ct in self.cellular_types:
            # Skip if at maximum population capacity
            if ct.x.size >= self.config.max_particles_per_type:
                continue

            # Validate array sizes to prevent indexing errors
            if not all(
                getattr(ct, attr).size == ct.x.size
                for attr in [
                    "y",
                    "vx",
                    "vy",
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
                ]
            ):
                continue

            # Identify components eligible for reproduction
            eligible: BoolArray = (
                ct.alive
                & (ct.energy > self.config.reproduction_energy_threshold)
                & (np.random.random(ct.x.size) < ct.reproduction_rate)
            )

            # Get count and indices of eligible parents
            num_offspring: int = int(np.sum(eligible))
            if num_offspring == 0:
                continue

            parent_indices: IntArray = np.where(eligible)[0]

            # Parents expend energy to reproduce
            ct.energy[eligible] *= 0.5
            parent_energy: FloatArray = ct.energy[eligible]

            # Calculate offspring energy share
            offspring_energy: FloatArray = (
                parent_energy * self.config.reproduction_offspring_energy_fraction
            )

            # Create mutation masks for different traits
            gene_mutation_mask: BoolArray = (
                np.random.random(num_offspring)
                < self.config.genetics.gene_mutation_rate
            )

            efficiency_mutation_mask: BoolArray = (
                np.random.random(num_offspring)
                < self.config.genetics.energy_efficiency_mutation_rate
            )

            # Inherit and mutate all genetic traits
            offspring_traits: Dict[str, FloatArray] = {
                "speed_factor": mutate_trait(
                    ct.speed_factor[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "interaction_strength": mutate_trait(
                    ct.interaction_strength[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "perception_range": mutate_trait(
                    ct.perception_range[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "reproduction_rate": mutate_trait(
                    ct.reproduction_rate[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "synergy_affinity": mutate_trait(
                    ct.synergy_affinity[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "colony_factor": mutate_trait(
                    ct.colony_factor[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "drift_sensitivity": mutate_trait(
                    ct.drift_sensitivity[parent_indices],
                    gene_mutation_mask,
                    self.config.genetics.gene_mutation_range[0],
                    self.config.genetics.gene_mutation_range[1],
                ),
                "energy_efficiency": mutate_trait(
                    ct.energy_efficiency[parent_indices],
                    efficiency_mutation_mask,
                    self.config.genetics.energy_efficiency_mutation_range[0],
                    self.config.genetics.energy_efficiency_mutation_range[1],
                ),
            }

            # Clamp all genetic values to valid ranges
            (
                offspring_traits["speed_factor"],
                offspring_traits["interaction_strength"],
                offspring_traits["perception_range"],
                offspring_traits["reproduction_rate"],
                offspring_traits["synergy_affinity"],
                offspring_traits["colony_factor"],
                offspring_traits["drift_sensitivity"],
            ) = self.config.genetics.clamp_gene_values(
                offspring_traits["speed_factor"],
                offspring_traits["interaction_strength"],
                offspring_traits["perception_range"],
                offspring_traits["reproduction_rate"],
                offspring_traits["synergy_affinity"],
                offspring_traits["colony_factor"],
                offspring_traits["drift_sensitivity"],
            )

            # Clamp energy efficiency to valid range
            offspring_traits["energy_efficiency"] = np.clip(
                offspring_traits["energy_efficiency"],
                self.config.energy_efficiency_range[0],
                self.config.energy_efficiency_range[1],
            )

            # Handle mass for mass-based types
            offspring_mass: Optional[FloatArray] = None
            if ct.mass_based and ct.mass is not None:
                offspring_mass = ct.mass[parent_indices].copy()
                mass_mutation_mask: BoolArray = np.random.random(num_offspring) < 0.1

                if np.any(mass_mutation_mask):
                    offspring_mass[mass_mutation_mask] *= np.random.uniform(
                        0.95, 1.05, size=int(np.sum(mass_mutation_mask))
                    )
                offspring_mass = np.maximum(offspring_mass, 0.1)

            # Calculate genetic distance for speciation
            genetic_distance: FloatArray = np.sqrt(
                np.sum(
                    [
                        (offspring_traits[trait] - getattr(ct, trait)[parent_indices])
                        ** 2
                        for trait in self.config.genetics.gene_traits
                    ],
                    axis=0,
                )
            )

            # Determine species IDs based on genetic distance threshold
            max_species_id: int = (
                int(np.max(ct.species_id)) if ct.species_id.size > 0 else ct.type_id
            )
            species_ids: IntArray = np.where(
                genetic_distance > self.config.speciation_threshold,
                max_species_id + 1,  # New species
                ct.species_id[parent_indices],  # Same species
            ).astype(np.int_)

            # Add all offspring to their cellular type
            for i in range(num_offspring):
                try:
                    # Calculate velocity based on efficiency and speed factor
                    velocity_scale: float = (
                        self.config.base_velocity_scale
                        / offspring_traits["energy_efficiency"][i]
                        * offspring_traits["speed_factor"][i]
                    )

                    # Add with small position offset from parent
                    ct.add_component(
                        x=float(ct.x[parent_indices[i]]) + np.random.uniform(-5, 5),
                        y=float(ct.y[parent_indices[i]]) + np.random.uniform(-5, 5),
                        vx=np.random.uniform(-0.5, 0.5) * velocity_scale,
                        vy=np.random.uniform(-0.5, 0.5) * velocity_scale,
                        energy=float(offspring_energy[i]),
                        mass_val=(
                            float(offspring_mass[i])
                            if offspring_mass is not None
                            else None
                        ),
                        energy_efficiency_val=float(
                            offspring_traits["energy_efficiency"][i]
                        ),
                        speed_factor_val=float(offspring_traits["speed_factor"][i]),
                        interaction_strength_val=float(
                            offspring_traits["interaction_strength"][i]
                        ),
                        perception_range_val=float(
                            offspring_traits["perception_range"][i]
                        ),
                        reproduction_rate_val=float(
                            offspring_traits["reproduction_rate"][i]
                        ),
                        synergy_affinity_val=float(
                            offspring_traits["synergy_affinity"][i]
                        ),
                        colony_factor_val=float(offspring_traits["colony_factor"][i]),
                        drift_sensitivity_val=float(
                            offspring_traits["drift_sensitivity"][i]
                        ),
                        species_id_val=int(species_ids[i]),
                        parent_id_val=int(parent_indices[i]),
                        max_age=ct.max_age,
                    )
                except IndexError:
                    # Skip if array indices are out of bounds
                    continue
