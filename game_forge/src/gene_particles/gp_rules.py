#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gene Particles Interaction Rules Module.

Provides sophisticated interaction mechanics between cellular types, implementing a
physics-based emergent behavior system with precise simulation boundaries.

The module implements three primary interaction systems:
    1. Force-based dynamics (potential, gravitational, and hybrid models)
    2. Energy transfer mechanisms (predator-prey relationships)
    3. Synergistic relationships (cooperative energy sharing)

All interactions use vectorized operations for performance and include stochastic
evolution parameters to create dynamic, emergent behaviors through simulation time.
"""

import random
from typing import List, Protocol, Tuple

import numpy as np
from eidosian_core import eidosian

from game_forge.src.gene_particles.gp_config import (
    EVOLUTION_ADJUSTMENT_RANGE,
    GIVE_TAKE_PROBABILITY,
    GRAVITY_STRENGTH_RANGE,
    INTERACTION_DISTANCE_RANGE,
    MIN_INTERACTION_DISTANCE,
    SYNERGY_PROBABILITY,
    SYNERGY_STRENGTH_RANGE,
    SimulationConfig,
)
from game_forge.src.gene_particles.gp_types import (
    BoolArray,
    FloatArray,
    InteractionParams,
)


class InteractionEvolver(Protocol):
    """Protocol defining how interaction parameters evolve over time.

    Implementers must provide a method to evolve their interaction parameters
    based on the current simulation frame count, creating temporal dynamics.
    """

    @eidosian()
    def evolve_parameters(self, frame_count: int) -> None:
        """Evolve interaction parameters over time.

        Parameters
        ----------
        frame_count : int
            Current simulation frame count
        """
        ...


class InteractionRules:
    """Manages creation and evolution of interaction parameters between cellular types.

    This class handles the complex relationships between different cellular types,
    including force-based interactions, energy transfer mechanisms (give-take),
    and cooperative energy sharing (synergy). All relationships evolve over time
    to create dynamic, emergent behaviors in the simulation.

    Attributes
    ----------
    config : SimulationConfig
        Configuration parameters for the simulation
    mass_based_type_indices : List[int]
        List of cellular type indices that use mass-based physics
    rules : List[Tuple[int, int, InteractionParams]]
        Matrix of interaction parameters between each cellular type pair
    give_take_matrix : BoolArray
        Boolean matrix defining energy transfer relationships
    synergy_matrix : FloatArray
        Float matrix defining energy sharing relationship strengths
    """

    def __init__(
        self, config: SimulationConfig, mass_based_type_indices: List[int]
    ) -> None:
        """Initialize the InteractionRules with configuration and mass-based type indices.

        Parameters
        ----------
        config : SimulationConfig
            Configuration parameters controlling simulation behavior
        mass_based_type_indices : List[int]
            Indices of cellular types that use mass-based physics
        """
        self.config: SimulationConfig = config
        self.mass_based_type_indices: List[int] = mass_based_type_indices
        self.rules: List[Tuple[int, int, InteractionParams]] = (
            self._create_interaction_matrix()
        )
        self.give_take_matrix: BoolArray = self._create_give_take_matrix()
        self.synergy_matrix: FloatArray = self._create_synergy_matrix()

    def _create_interaction_matrix(self) -> List[Tuple[int, int, InteractionParams]]:
        """Create a matrix of interaction parameters for all cellular type pairs.

        For each pair of cellular types (i,j), generates a parameter dictionary
        defining how they interact through forces, with specialized handling
        for mass-based types.

        Returns
        -------
        List[Tuple[int, int, InteractionParams]]
            List of tuples (i, j, params) where i and j are cellular type indices
            and params is a dictionary of interaction parameters
        """
        n_types: int = self.config.n_cell_types
        final_rules: List[Tuple[int, int, InteractionParams]] = []

        # Generate unique physics parameters for each cell type pair
        for i in range(n_types):
            for j in range(n_types):
                params: InteractionParams = self._random_interaction_params(i, j)
                final_rules.append((i, j, params))

        return final_rules

    def _random_interaction_params(self, i: int, j: int) -> InteractionParams:
        """Generate randomized interaction parameters between two cellular types.

        Creates a physics-based interaction model between cellular types i and j,
        with potential-based and/or gravity-based forces depending on their
        mass characteristics.

        Parameters
        ----------
        i : int
            Index of the first cellular type
        j : int
            Index of the second cellular type

        Returns
        -------
        InteractionParams
            Dictionary of interaction parameters defining force calculations
        """
        # Determine if both types are mass-based for gravity interactions
        both_mass: bool = (
            i in self.mass_based_type_indices and j in self.mass_based_type_indices
        )

        # 50% chance to use gravity if both types are mass-based
        use_gravity: bool = both_mass and (random.random() < 0.5)

        # Always use potential-based interactions
        use_potential: bool = True

        # Generate random potential strength within configured range
        potential_strength: float = random.uniform(
            self.config.interaction_strength_range[0],
            self.config.interaction_strength_range[1],
        )

        # 50% chance to invert the potential (attraction vs repulsion)
        if random.random() < 0.5:
            potential_strength = -potential_strength

        # Set gravity factor if applicable, otherwise zero
        gravity_factor: float = (
            random.uniform(GRAVITY_STRENGTH_RANGE[0], GRAVITY_STRENGTH_RANGE[1])
            if use_gravity
            else 0.0
        )

        # Random interaction distance within standard bounds
        max_dist: float = random.uniform(
            INTERACTION_DISTANCE_RANGE[0], INTERACTION_DISTANCE_RANGE[1]
        )

        # Compile interaction parameters into strongly-typed structure
        params: InteractionParams = {
            "use_potential": use_potential,
            "use_gravity": use_gravity,
            "potential_strength": potential_strength,
            "gravity_factor": gravity_factor,
            "max_dist": max_dist,
        }

        return params

    def _create_give_take_matrix(self) -> BoolArray:
        """Create a matrix defining predator-prey energy transfer relationships.

        Generates a sparse boolean matrix where element (i,j) is True if
        cellular type i transfers energy to cellular type j during close
        encounters (prey-predator relationship).

        Returns
        -------
        BoolArray
            Boolean matrix of shape (n_cell_types, n_cell_types) defining
            give-take relationships between cellular types
        """
        n_types: int = self.config.n_cell_types

        # Initialize empty relationship matrix
        matrix: BoolArray = np.zeros((n_types, n_types), dtype=bool)

        # For each non-identical cellular type pair
        for i in range(n_types):
            for j in range(n_types):
                if i != j:
                    # Establish give-take relationship with defined probability
                    if random.random() < GIVE_TAKE_PROBABILITY:
                        matrix[i, j] = True

        return matrix

    def _create_synergy_matrix(self) -> FloatArray:
        """Create a matrix defining cooperative energy sharing relationships.

        Generates a sparse float matrix where element (i,j) represents the
        strength of energy sharing from type i to type j when they interact
        within synergy range. Higher values mean stronger sharing.

        Returns
        -------
        FloatArray
            Float matrix of shape (n_cell_types, n_cell_types) with values
            in range [0.0, 1.0] representing synergy strengths
        """
        n_types: int = self.config.n_cell_types

        # Initialize empty synergy matrix
        synergy_matrix: FloatArray = np.zeros((n_types, n_types), dtype=np.float64)

        # For each non-identical cellular type pair
        for i in range(n_types):
            for j in range(n_types):
                if i != j:
                    # Establish synergy relationship with defined probability
                    if random.random() < SYNERGY_PROBABILITY:
                        # Assign random synergy factor within standard range
                        # Higher values create stronger energy sharing
                        synergy_matrix[i, j] = random.uniform(
                            SYNERGY_STRENGTH_RANGE[0], SYNERGY_STRENGTH_RANGE[1]
                        )
                    else:
                        # No synergy relationship
                        synergy_matrix[i, j] = 0.0

        return synergy_matrix

    @eidosian()
    def evolve_parameters(self, frame_count: int) -> None:
        """Evolve interaction parameters periodically to create dynamic relationships.

        Mutates interaction forces, energy transfer efficiencies, and synergy
        relationships at regular intervals defined by evolution_interval,
        creating evolutionary pressure and emergent behaviors.

        Parameters
        ----------
        frame_count : int
            Current simulation frame number, used to determine
            when evolution should occur
        """
        # Only evolve at specified frame intervals
        if frame_count % self.config.evolution_interval != 0:
            return

        self._evolve_interaction_forces()
        self._evolve_energy_transfer()
        self._evolve_synergy_relationships()

    def _evolve_interaction_forces(self) -> None:
        """Evolve physical interaction forces with small random adjustments.

        Each force parameter has a small chance to mutate slightly, creating
        gradually changing dynamics while maintaining simulation stability.
        """
        for _, _, params in self.rules:
            # 10% chance to adjust potential strength by ±5%
            if random.random() < 0.1:
                params["potential_strength"] *= random.uniform(
                    EVOLUTION_ADJUSTMENT_RANGE[0], EVOLUTION_ADJUSTMENT_RANGE[1]
                )

            # 5% chance to adjust gravity factor if present
            if random.random() < 0.05 and params["use_gravity"]:
                params["gravity_factor"] *= random.uniform(
                    EVOLUTION_ADJUSTMENT_RANGE[0], EVOLUTION_ADJUSTMENT_RANGE[1]
                )

            # 5% chance to adjust interaction distance
            if random.random() < 0.05:
                # Ensure distance stays above minimum threshold
                params["max_dist"] = max(
                    MIN_INTERACTION_DISTANCE,
                    params["max_dist"]
                    * random.uniform(
                        EVOLUTION_ADJUSTMENT_RANGE[0], EVOLUTION_ADJUSTMENT_RANGE[1]
                    ),
                )

    def _evolve_energy_transfer(self) -> None:
        """Evolve energy transfer efficiency across the simulation.

        Small random adjustments to global energy transfer rate create
        evolving predator-prey dynamics while maintaining physical plausibility.
        """
        # 10% chance to evolve global energy transfer efficiency
        if random.random() < 0.1:
            self.config.energy_transfer_factor = min(
                1.0,  # Maximum transfer efficiency cap
                self.config.energy_transfer_factor
                * random.uniform(
                    EVOLUTION_ADJUSTMENT_RANGE[0], EVOLUTION_ADJUSTMENT_RANGE[1]
                ),
            )

    def _evolve_synergy_relationships(self) -> None:
        """Evolve synergy relationships between cell types.

        Each synergy relationship has a chance to strengthen or weaken based on
        the global evolution rate, creating shifting cooperative dynamics.
        """
        # Evolve synergy relationships based on configured rate
        n_types: int = self.synergy_matrix.shape[0]  # Number of cell types
        for i in range(n_types):
            for j in range(n_types):
                if random.random() < self.config.synergy_evolution_rate:
                    # Apply small random adjustment between -0.05 and +0.05
                    adjustment: float = random.random() * 0.1 - 0.05

                    # Adjust synergy factor, keeping it within valid range [0.0, 1.0]
                    self.synergy_matrix[i, j] = min(
                        1.0, max(0.0, self.synergy_matrix[i, j] + adjustment)
                    )
