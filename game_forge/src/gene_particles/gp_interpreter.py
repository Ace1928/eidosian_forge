"""Gene Particles Genetic Interpreter Module.

Provides the core genetic interpretation mechanics, translating genotypic data into
phenotypic expressions, mutation processes, and behavioral traits with precise
static typing and vectorized operations for emergent evolutionary dynamics.

The interpreter serves as a biomimetic decoder that maps symbolic genetic sequences
to functional behaviors, allowing for hereditary trait transmission, evolutionary
adaptation, and complex emergent phenomena through vectorized particle operations.
"""

from typing import TYPE_CHECKING, List, Optional, cast
from eidosian_core import eidosian

# Use TYPE_CHECKING for circular imports
if TYPE_CHECKING:
    from game_forge.src.gene_particles.gp_config import SimulationConfig
    from game_forge.src.gene_particles.gp_types import CellularTypeData

from game_forge.src.gene_particles.gp_config import ReproductionMode
from game_forge.src.gene_particles.gp_genes import (
    apply_energy_gene,
    apply_growth_gene,
    apply_interaction_gene,
    apply_movement_gene,
    apply_predation_gene,
    apply_reproduction_gene,
)
from game_forge.src.gene_particles.gp_types import GeneData, GeneSequence

###############################################################
# Genetic Interpreter Class
###############################################################


class GeneticInterpreter:
    """Decodes and interprets the genetic sequence of particles.

    Translates genetic data into behavioral traits, reproduction mechanics,
    and inter-particle interaction dynamics through a biomimetic interpretation
    system. Each gene contains redundant components and multiple mutable
    elements for evolutionary resilience, enabling complex phenotypic expression
    through simple symbolic encoding.

    Attributes:
        default_sequence: Default genetic blueprint if none provided
        gene_sequence: Active genetic data used for particle behavior
    """

    def __init__(self, gene_sequence: Optional[GeneSequence] = None) -> None:
        """Initialize the GeneticInterpreter with a gene sequence.

        Args:
            gene_sequence: Encoded genetic data as a list of gene arrays.
                If None, uses default sequence with baseline behaviors.
        """
        # Default gene sequence if none provided - serves as genetic baseline
        self.default_sequence: GeneSequence = [
            # [gene_type, param1, param2, param3, ...]
            # Format: [speed_modifier, randomness, direction_bias]
            ["start_movement", 1.0, 0.1, 0.0],
            # Format: [attraction_strength, interaction_radius]
            ["start_interaction", 0.5, 100.0],
            # Format: [passive_gain, feeding_efficiency, predation_efficiency]
            ["start_energy", 0.1, 0.5, 0.3],
            # Format: [sexual_threshold, asexual_threshold, reproduction_cost, cooldown_time]
            [
                "start_reproduction",
                150.0,
                100.0,
                50.0,
                30.0,
                1.0,
                0.65,
                1.0,
            ],
            # Format: [growth_rate, adult_size, maturity_age]
            ["start_growth", 0.1, 2.0, 100.0],
            # Format: [attack_power, energy_conversion, predation_strategy, detection_range]
            ["start_predation", 1.0, 0.5, 0.0, 100.0],
        ]
        self.gene_sequence: GeneSequence = (
            gene_sequence if gene_sequence is not None else self.default_sequence
        )
        self.apply_predation_gene = apply_predation_gene
        self.apply_growth_gene = apply_growth_gene
        self.apply_energy_gene = apply_energy_gene
        self.apply_interaction_gene = apply_interaction_gene
        self.apply_movement_gene = apply_movement_gene
        self.apply_reproduction_gene = apply_reproduction_gene
        self._dispatch = {
            "start_movement": self.apply_movement_gene,
            "start_interaction": self.apply_interaction_gene,
            "start_energy": self.apply_energy_gene,
            "start_growth": self.apply_growth_gene,
            "start_predation": self.apply_predation_gene,
            "start_reproduction": self.apply_reproduction_gene,
        }

    @eidosian()
    def decode(
        self,
        particle: "CellularTypeData",
        others: List["CellularTypeData"],
        env: "SimulationConfig",
    ) -> None:
        """Decode genetic sequence to influence particle behavior.

        Maps each gene's symbolic structure to corresponding traits and actions,
        applying the genetic blueprint to determine particle behavior in the
        simulation environment through precise phenotypic expression.

        Args:
            particle: Cellular type data of the particle to decode. Contains all
                phenotypic expressions, state variables, and behavioral traits.
            others: List of other cellular types for interaction calculations.
                Used for social behaviors, predation mechanics, and reproductive pairing.
            env: Environmental configuration parameters affecting gene expression.
                Controls how genes manifest under different simulation conditions.
        """
        # Iterate through each gene in the gene sequence
        for gene in self.gene_sequence:
            if not gene or len(gene) < 2:
                continue  # Skip invalid genes (need at least type + one parameter)

            gene_type = cast(str, gene[0])  # First element is always the gene type
            gene_data = [
                float(val) for val in gene[1:]
            ]  # Convert parameters to float for vectorized operations

            try:
                # Route gene to appropriate handler method using dispatch pattern
                self._route_gene_to_handler(gene_type, particle, others, gene_data, env)
            except Exception as e:
                # Log error but continue processing other genes (fault tolerance)
                print(f"Error processing gene {gene_type}: {str(e)}")

    def _route_gene_to_handler(
        self,
        gene_type: str,
        particle: "CellularTypeData",
        others: List["CellularTypeData"],
        gene_data: GeneData,
        env: "SimulationConfig",
    ) -> None:
        """Route gene to appropriate handler based on type.

        A dispatcher pattern for enhanced performance and maintainability,
        mapping gene types to their handler methods.

        Args:
            gene_type: The type identifier of the gene
            particle: Cellular type data of the particle
            others: List of other cellular types for interaction
            gene_data: Decoded numeric parameters for the gene
            env: Environmental configuration parameters
        """
        handler = self._dispatch.get(gene_type)
        if handler is None:
            return

        if gene_type == "start_reproduction":
            if env.reproduction_mode in (ReproductionMode.GENES, ReproductionMode.HYBRID):
                self.apply_reproduction_gene(particle, others, gene_data, env)
            return

        if gene_type in {"start_interaction", "start_predation"}:
            handler(particle, others, gene_data, env)
        else:
            handler(particle, gene_data, env)
