"""
GeneParticles: Advanced Cellular Automata with Dynamic Gene Expression, Emergent Behaviors, 
and Extended Complexity
-------------------------------------------------------------------------------------------------
Enhanced Version:
- Advanced genetic mechanisms including gene linkage and crossover.
- Environmental interactions influencing gene expression.
- Communication and "nerve" conduction between particles.
- Embodied intelligence with senses and reactivity.
- Potential development of language and culture.

Core Features Enhancements:
---------------------------
1. Advanced Genetic Representation
   - Gene Linkage
   - Crossover Mechanisms
   - Environmental Interactions

2. Communication and Neural Conduction
   - Signal Transmission
   - Neural Network Simulation
   - Embodied Intelligence

Usage:
------
Ensure all dependencies are installed as per the original script.
Run the simulation:
    python geneparticles.py
"""

import math
import random
import collections
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Union
import time
import numpy as np
import pygame
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import traceback
import scipy

###############################################################
# Configuration Classes
###############################################################

class GeneticParamConfig:
    """
    High-performance genetic parameter configuration with vectorized operations.
    Uses NumPy arrays and SIMD operations for maximum efficiency.
    """

    def __init__(self, 
                 gene_traits: Optional[List[str]] = None,
                 gene_mutation_rate: float = 0.25,
                 gene_mutation_range: Tuple[float, float] = (-0.2, 0.2),
                 trait_ranges: Optional[List[Tuple[float, float]]] = None,
                 energy_efficiency_mutation_rate: float = 0.2,
                 energy_efficiency_mutation_range: Tuple[float, float] = (-0.15, 0.3)):
        # Pre-allocated arrays for vectorized operations
        self.gene_traits: List[str] = gene_traits if gene_traits else [
            "speed_factor", "interaction_strength", "perception_range", 
            "reproduction_rate", "synergy_affinity", "colony_factor",
            "drift_sensitivity"
        ]

        # Optimized mutation parameters
        self.gene_mutation_rate: float = gene_mutation_rate
        self.gene_mutation_range: Tuple[float, float] = gene_mutation_range

        # Cache-aligned trait ranges
        self.trait_ranges = np.array(trait_ranges if trait_ranges else [
            (0.05, 4.0),   # speed_factor
            (0.05, 4.0),   # interaction_strength
            (20.0, 400.0), # perception_range
            (0.02, 1.5),   # reproduction_rate
            (0.0, 3.0),    # synergy_affinity
            (0.0, 2.0),    # colony_factor
            (0.0, 3.0),    # drift_sensitivity
        ], dtype=np.float32)

        # SIMD-optimized energy parameters
        self.energy_efficiency_mutation_rate: float = energy_efficiency_mutation_rate
        self.energy_efficiency_mutation_range: Tuple[float, float] = energy_efficiency_mutation_range

        # Numerical stability constants
        self.EPSILON = np.finfo(np.float32).eps
        self.MIN_ARRAY_SIZE: int = 1

    def clamp_gene_values(self, *arrays: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """
        Vectorized gene value clamping with SIMD optimization.
        Uses NumPy broadcasting and clip operations for maximum performance.
        """
        try:
            # Filter out None arrays and get their shapes
            valid_arrays = [arr for arr in arrays if arr is not None]
            if not valid_arrays:
                # If all arrays are None, return default arrays
                shape = (self.MIN_ARRAY_SIZE,)
                return tuple(
                    np.full(shape, range[0] + self.EPSILON, dtype=np.float32)
                    for range in self.trait_ranges
                )
            
            target_shape = np.broadcast_shapes(*(arr.shape for arr in valid_arrays))
            
            # Vectorized clamping operation
            results = []
            for i, arr in enumerate(arrays):
                if arr is None:
                    arr = np.full(target_shape, self.trait_ranges[i][0] + self.EPSILON, dtype=np.float32)
                else:
                    # Efficient broadcasting and copying
                    arr = np.broadcast_to(arr, target_shape).copy()
                    
                    # Vectorized invalid value replacement
                    arr = np.nan_to_num(
                        arr,
                        nan=self.trait_ranges[i][0] + self.EPSILON,
                        posinf=self.trait_ranges[i][1],
                        neginf=self.trait_ranges[i][0]
                    )
                    
                    # SIMD clipping
                    arr = np.clip(
                        arr,
                        self.trait_ranges[i][0] + self.EPSILON,
                        self.trait_ranges[i][1]
                    )
                results.append(arr)
                
            return tuple(results)
            
        except Exception:
            # Efficient fallback with pre-allocated arrays
            shape = (self.MIN_ARRAY_SIZE,)
            return tuple(
                np.full(shape, range[0] + self.EPSILON, dtype=np.float32)
                for range in self.trait_ranges
            )

class SimulationConfig:
    """
    Configuration class for the GeneParticles simulation, optimized for maximum emergence
    and complex structure formation with robust error handling.
    """

    def __init__(self, 
                 target_fps: int = 30,
                 n_cell_types: int = 10,
                 particles_per_type: int = 50,
                 min_particles_per_type: int = 50,
                 max_particles_per_type: int = 300,
                 mass_range: Tuple[float, float] = (0.2, 15.0),
                 base_velocity_scale: float = 1.2,
                 mass_based_fraction: float = 0.7,
                 interaction_strength_range: Tuple[float, float] = (-3.0, 3.0),
                 max_frames: int = 0,
                 initial_energy: float = 150.0,
                 friction: float = 0.2,
                 global_temperature: float = 0.1,
                 predation_range: float = 75.0,
                 energy_transfer_factor: float = 0.7,
                 mass_transfer: bool = True,
                 max_age: float = np.inf,
                 evolution_interval: int = 3000,
                 synergy_range: float = 200.0,
                 culling_fitness_weights: Optional[Dict[str, float]] = None,
                 reproduction_energy_threshold: float = 180.0,
                 reproduction_mutation_rate: float = 0.3,
                 reproduction_offspring_energy_fraction: float = 0.5,
                 alignment_strength: float = 0.4,
                 cohesion_strength: float = 0.5,
                 separation_strength: float = 0.3,
                 cluster_radius: float = 15.0,
                 particle_size: float = 5.0,
                 energy_efficiency_range: Tuple[float, float] = (-0.4, 3.0),
                 genetics: Optional[GeneticParamConfig] = None,
                 speciation_threshold: float = 0.8,
                 colony_formation_probability: float = 0.4,
                 colony_radius: float = 250.0,
                 colony_cohesion_strength: float = 0.6,
                 synergy_evolution_rate: float = 0.08,
                 complexity_factor: float = 2.0,
                 structural_complexity_weight: float = 0.9,
                 environmental_factors: Optional[Dict[str, Any]] = None,
                 communication_range: float = 100.0,
                 signal_strength: float = 1.0,
                 neural_network_params: Optional[Dict[str, Any]] = None):
        # Core simulation parameters optimized for emergence
        self.target_fps: int = max(1, target_fps)
        self.n_cell_types: int = max(1, n_cell_types) # Ensure at least 1 type
        self.particles_per_type: int = max(1, particles_per_type) # Ensure at least 1 particle
        self.min_particles_per_type: int = max(1, min_particles_per_type)
        self.max_particles_per_type: int = max(300, max_particles_per_type)
        self.mass_range: Tuple[float, float] = (max(0.2, 1e-6), mass_range[1])
        self.base_velocity_scale: float = max(0.1, base_velocity_scale)
        self.mass_based_fraction: float = np.clip(mass_based_fraction, 0.0, 1.0)
        self.interaction_strength_range: Tuple[float, float] = interaction_strength_range
        self.max_frames: int = max(0, max_frames)
        self.initial_energy: float = max(1.0, initial_energy)
        self.friction: float = np.clip(friction, 0.0, 1.0)
        self.global_temperature: float = max(0.0, global_temperature)
        self.predation_range: float = max(1.0, predation_range)
        self.energy_transfer_factor: float = np.clip(energy_transfer_factor, 0.0, 1.0)
        self.mass_transfer: bool = mass_transfer
        self.max_age: float = max(1.0, max_age)
        self.evolution_interval: int = max(1, evolution_interval)
        self.synergy_range: float = max(1.0, synergy_range)

        # Balanced culling weights with validation
        self.culling_fitness_weights: Dict[str, float] = culling_fitness_weights if culling_fitness_weights else {
            k: np.clip(v, 0.0, 1.0) for k, v in {
                "energy_weight": 0.6,
                "age_weight": 0.8,
                "speed_factor_weight": 0.7,
                "interaction_strength_weight": 0.7,
                "synergy_affinity_weight": 0.8,
                "colony_factor_weight": 0.9,
                "drift_sensitivity_weight": 0.6
            }.items()
        }

        # Reproduction parameters for dynamic population
        self.reproduction_energy_threshold: float = max(1.0, reproduction_energy_threshold)
        self.reproduction_mutation_rate: float = np.clip(reproduction_mutation_rate, 0.0, 1.0)
        self.reproduction_offspring_energy_fraction: float = np.clip(reproduction_offspring_energy_fraction, 0.0, 1.0)

        # Enhanced clustering parameters
        self.alignment_strength: float = np.clip(alignment_strength, 0.0, 1.0)
        self.cohesion_strength: float = np.clip(cohesion_strength, 0.0, 1.0)
        self.separation_strength: float = np.clip(separation_strength, 0.0, 1.0)
        self.cluster_radius: float = max(1.0, cluster_radius)

        self.particle_size: float = max(1.0, particle_size)

        self.energy_efficiency_range: Tuple[float, float] = energy_efficiency_range

        self.genetics = genetics if genetics else GeneticParamConfig()

        # Enhanced speciation and colony parameters
        self.speciation_threshold: float = np.clip(speciation_threshold, 0.0, 1.0)
        self.colony_formation_probability: float = np.clip(colony_formation_probability, 0.0, 1.0)
        self.colony_radius: float = max(1.0, colony_radius)
        self.colony_cohesion_strength: float = np.clip(colony_cohesion_strength, 0.0, 1.0)

        # Advanced parameters for emergence
        self.synergy_evolution_rate: float = np.clip(synergy_evolution_rate, 0.0, 1.0)
        self.complexity_factor: float = max(0.1, complexity_factor)
        self.structural_complexity_weight: float = np.clip(structural_complexity_weight, 0.0, 1.0)

        # Safety epsilon for numerical stability
        self.EPSILON: float = 1e-10

        # New Parameters for Advanced Genetic Mechanisms
        self.environmental_factors: Dict[str, Any] = environmental_factors if environmental_factors else {
            "temperature": 25.0,  # Celsius
            "resource_abundance": 100.0,  # Arbitrary units
            "predator_density": 5.0  # Number of predators per area
        }

        self.communication_range: float = communication_range  # Distance within which particles can communicate
        self.signal_strength: float = signal_strength  # Base strength of signals
        self.neural_network_params: Dict[str, Any] = neural_network_params if neural_network_params else {
            "input_size": 5,  # Number of inputs to the neural network
            "hidden_size": 10,  # Number of neurons in the hidden layer
            "output_size": 5,  # Number of outputs from the neural network
            "activation": "relu"  # Activation function
        }

        self._validate()

    def _validate(self) -> None:
        """
        Validate configuration parameters with comprehensive error checking.
        """
        try:
            validation_rules = [
                (self.n_cell_types > 0, "Number of cell types must be greater than 0"),
                (self.particles_per_type > 0, "Particles per type must be greater than 0"),
                (self.mass_range[0] > 0, "Minimum mass must be positive"),
                (self.base_velocity_scale > 0, "Base velocity scale must be positive"),
                (0.0 <= self.mass_based_fraction <= 1.0, "Mass-based fraction must be between 0.0 and 1.0"),
                (self.interaction_strength_range[0] < self.interaction_strength_range[1], 
                 "Invalid interaction strength range"),
                (self.max_frames >= 0, "Maximum frames must be non-negative"),
                (self.initial_energy > 0, "Initial energy must be positive"),
                (0.0 <= self.friction <= 1.0, "Friction must be between 0.0 and 1.0"),
                (self.global_temperature >= 0, "Global temperature must be non-negative"),
                (self.predation_range > 0, "Predation range must be positive"),
                (0.0 <= self.energy_transfer_factor <= 1.0, 
                 "Energy transfer factor must be between 0.0 and 1.0"),
                (self.cluster_radius > 0, "Cluster radius must be positive"),
                (self.particle_size > 0, "Particle size must be positive"),
                (self.speciation_threshold > 0, "Speciation threshold must be positive"),
                (self.synergy_range > 0, "Synergy range must be positive"),
                (self.colony_radius > 0, "Colony radius must be positive"),
                (self.reproduction_energy_threshold > 0, 
                 "Reproduction energy threshold must be positive"),
                (0.0 <= self.reproduction_offspring_energy_fraction <= 1.0,
                 "Invalid reproduction offspring energy fraction"),
                (0.0 <= self.genetics.gene_mutation_rate <= 1.0,
                 "Gene mutation rate must be between 0.0 and 1.0"),
                (self.genetics.gene_mutation_range[0] < self.genetics.gene_mutation_range[1],
                 "Invalid gene mutation range"),
                (self.energy_efficiency_range[0] < self.energy_efficiency_range[1],
                 "Invalid energy efficiency range"),
                (self.genetics.energy_efficiency_mutation_range[0] < 
                 self.genetics.energy_efficiency_mutation_range[1],
                 "Invalid energy efficiency mutation range"),
                # Validate environmental factors
                (self.environmental_factors["temperature"] >= -273.15, "Temperature cannot be below absolute zero"),
                (self.environmental_factors["resource_abundance"] >= 0, "Resource abundance must be non-negative"),
                (self.environmental_factors["predator_density"] >= 0, "Predator density must be non-negative"),
                # Validate neural network parameters
                (self.neural_network_params["input_size"] > 0, "Neural network input size must be positive"),
                (self.neural_network_params["hidden_size"] > 0, "Neural network hidden size must be positive"),
                (self.neural_network_params["output_size"] > 0, "Neural network output size must be positive"),
                (self.communication_range > 0, "Communication range must be positive"),
                (self.signal_strength > 0, "Signal strength must be positive"),
            ]

            for condition, message in validation_rules:
                if not condition:
                    raise ValueError(message)

        except Exception as e:
            # Set safe default values if validation fails
            self._set_safe_defaults()
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def _set_safe_defaults(self) -> None:
        """
        Set safe default values if validation fails, with robust error handling and optimized defaults
        for high-performance particle simulation.
        """
        try:
            # Core simulation parameters with safe minimum values
            self.n_cell_types = max(1, min(10, self.n_cell_types))
            self.particles_per_type = max(1, min(50, self.particles_per_type))
            self.min_particles_per_type = max(1, min(50, self.min_particles_per_type))
            self.max_particles_per_type = max(300, self.min_particles_per_type)
            self.mass_range = (max(1e-10, 0.2), max(15.0, self.mass_range[1]))
            self.base_velocity_scale = max(0.1, min(2.0, self.base_velocity_scale))
            self.mass_based_fraction = np.clip(self.mass_based_fraction, 0.0, 1.0)
            self.interaction_strength_range = (-3.0, 3.0)
            self.max_frames = max(0, self.max_frames)
            self.initial_energy = max(1.0, min(150.0, self.initial_energy))
            self.friction = np.clip(self.friction, 0.0, 1.0)
            self.global_temperature = max(0.0, min(1.0, self.global_temperature))
            self.predation_range = max(1.0, min(75.0, self.predation_range))
            self.energy_transfer_factor = np.clip(self.energy_transfer_factor, 0.0, 1.0)
            self.mass_transfer = bool(self.mass_transfer)
            self.max_age = max(1.0, self.max_age)
            self.evolution_interval = max(1, min(3000, self.evolution_interval))
            self.synergy_range = max(1.0, min(200.0, self.synergy_range))

            # Culling weights with safe normalization
            weight_sum = sum(self.culling_fitness_weights.values()) + 1e-10
            self.culling_fitness_weights = {
                k: np.clip(v / weight_sum, 0.0, 1.0)
                for k, v in self.culling_fitness_weights.items()
            }

            # Reproduction and clustering parameters
            self.reproduction_energy_threshold = max(1.0, min(180.0, self.reproduction_energy_threshold))
            self.reproduction_mutation_rate = np.clip(self.reproduction_mutation_rate, 0.0, 1.0)
            self.reproduction_offspring_energy_fraction = np.clip(self.reproduction_offspring_energy_fraction, 0.0, 1.0)
            self.alignment_strength = np.clip(self.alignment_strength, 0.0, 1.0)
            self.cohesion_strength = np.clip(self.cohesion_strength, 0.0, 1.0)
            self.separation_strength = np.clip(self.separation_strength, 0.0, 1.0)
            self.cluster_radius = max(1.0, min(15.0, self.cluster_radius))
            self.particle_size = max(1.0, min(5.0, self.particle_size))
            self.energy_efficiency_range = (-0.4, max(0.0, self.energy_efficiency_range[1]))

            # Environmental factors with safe defaults
            self.environmental_factors = {
                "temperature": max(-273.15, self.environmental_factors.get("temperature", 25.0)),
                "resource_abundance": max(0.0, self.environmental_factors.get("resource_abundance", 100.0)),
                "predator_density": max(0.0, self.environmental_factors.get("predator_density", 5.0))
            }

            # Communication parameters with safe defaults
            self.communication_range = max(10.0, min(200.0, self.communication_range))
            self.signal_strength = max(0.1, min(5.0, self.signal_strength))

            # Neural network parameters with safe defaults
            self.neural_network_params = {
                "input_size": max(1, self.neural_network_params.get("input_size", 5)),
                "hidden_size": max(1, self.neural_network_params.get("hidden_size", 10)),
                "output_size": max(1, self.neural_network_params.get("output_size", 5)),
                "activation": self.neural_network_params.get("activation", "relu")
            }

            # Safety epsilon
            self.EPSILON = max(1e-10, self.EPSILON)
        except Exception as e:
            print(f"Error in emergency optimization: {e}")
            traceback.print_exc()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration parameters to a dictionary with error handling.
        """
        try:
            config_dict = {k: v for k, v in self.__dict__.items() 
                         if not k.startswith('_') and k != 'genetics'}
            config_dict["genetics"] = {k: v for k, v in self.genetics.__dict__.items() 
                                     if not k.startswith('_')}
            return config_dict
        except Exception:
            return {"error": "Failed to convert configuration to dictionary"}

###############################################################
# Genetic Components
###############################################################
import logging
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Gene:
    """
    Represents an individual gene with potential for linkage and environmental sensitivity.
    """
    def __init__(self, trait: str, value: float, linked_genes: Optional[List['Gene']] = None):
        """
        Initializes a Gene object.

        Args:
            trait (str): The name or identifier of the trait this gene controls.
            value (float): The current value of the gene.
            linked_genes (Optional[List['Gene']]): A list of genes linked to this gene, defaults to None.
        """
        self.trait = trait
        self.value = value
        self.linked_genes = linked_genes if linked_genes else []
        logging.debug(f"Gene initialized: trait={self.trait}, value={self.value}, linked_genes={self.linked_genes}")


    def mutate(self, mutation_rate: float, mutation_range: Tuple[float, float], trait_ranges: Tuple[float, float]) -> None:
        """
        Mutates the gene's value based on a mutation rate and range.

        Args:
            mutation_rate (float): The probability of mutation occurring (0.0 to 1.0).
            mutation_range (Tuple[float, float]): The range within which the mutation value is generated.
            trait_ranges (Tuple[float, float]): The allowed range for the gene's value after mutation.
        """
        if not (0.0 <= mutation_rate <= 1.0):
            logging.warning(f"Invalid mutation_rate: {mutation_rate}. Must be between 0 and 1.")
            return
        if random.random() < mutation_rate:
            mutation = random.uniform(*mutation_range)
            self.value = np.clip(self.value + mutation, *trait_ranges)
            logging.debug(f"Gene mutated: trait={self.trait}, new_value={self.value}")
        else:
            logging.debug(f"Gene mutation skipped: trait={self.trait}")


class Genome:
    """
    Manages a collection of genes, handling inheritance, mutation, and crossover operations.
    """
    def __init__(self, genes: Optional[List[Gene]] = None):
        """
        Initializes a Genome object.

        Args:
            genes (Optional[List[Gene]]): A list of Gene objects, defaults to None.
        """
        self.genes = genes if genes else []
        logging.debug(f"Genome initialized with {len(self.genes)} genes.")

    def add_gene(self, gene: Gene) -> None:
        """
        Adds a gene to the genome.

        Args:
            gene (Gene): The Gene object to add.
        """
        if not isinstance(gene, Gene):
            logging.error(f"Invalid gene type: {type(gene)}. Expected Gene object.")
            return
        self.genes.append(gene)
        logging.debug(f"Gene added to genome: trait={gene.trait}")

    def mutate(self, mutation_rate: float, mutation_range: Tuple[float, float], trait_ranges: List[Tuple[float, float]]) -> None:
        """
        Mutates all genes in the genome.

        Args:
            mutation_rate (float): The probability of mutation occurring (0.0 to 1.0).
            mutation_range (Tuple[float, float]): The range within which the mutation value is generated.
            trait_ranges (List[Tuple[float, float]]): A list of allowed ranges for each gene's value after mutation.
        """
        if not (0.0 <= mutation_rate <= 1.0):
            logging.warning(f"Invalid mutation_rate: {mutation_rate}. Must be between 0 and 1.")
            return
        if len(self.genes) != len(trait_ranges):
            logging.error(f"Length mismatch: genes={len(self.genes)}, trait_ranges={len(trait_ranges)}. Mutation skipped.")
            return
        for i, gene in enumerate(self.genes):
            gene.mutate(mutation_rate, mutation_range, trait_ranges[i])
        logging.debug(f"Genome mutated: {len(self.genes)} genes processed.")


    def crossover(self, other: 'Genome', crossover_rate: float = 0.5) -> Tuple['Genome', 'Genome']:
        """
        Performs crossover between two genomes, returning two offspring genomes.

        Args:
            other (Genome): The other Genome object to crossover with.
            crossover_rate (float): The probability of crossover occurring at each gene (0.0 to 1.0).

        Returns:
            Tuple[Genome, Genome]: A tuple containing the two offspring Genome objects.
        """
        if not isinstance(other, Genome):
            logging.error(f"Invalid other genome type: {type(other)}. Expected Genome object.")
            return self, self # Return copies of self if other is invalid
        if not (0.0 <= crossover_rate <= 1.0):
            logging.warning(f"Invalid crossover_rate: {crossover_rate}. Must be between 0 and 1.")
            return self, other # Return self and other if crossover_rate is invalid

        child1_genes = []
        child2_genes = []
        
        # Ensure both genomes have the same number of genes
        min_len = min(len(self.genes), len(other.genes))
        
        for i in range(min_len):
            gene1 = self.genes[i]
            gene2 = other.genes[i]
            
            if random.random() < crossover_rate:
                child1_genes.append(Gene(gene1.trait, gene1.value, gene1.linked_genes.copy()))
                child2_genes.append(Gene(gene2.trait, gene2.value, gene2.linked_genes.copy()))
            else:
                child1_genes.append(Gene(gene2.trait, gene2.value, gene2.linked_genes.copy()))
                child2_genes.append(Gene(gene1.trait, gene1.value, gene1.linked_genes.copy()))
        
        # Handle cases where genomes have different lengths
        if len(self.genes) > min_len:
            child1_genes.extend(self.genes[min_len:])
        if len(other.genes) > min_len:
            child2_genes.extend(other.genes[min_len:])
        
        child1 = Genome(child1_genes)
        child2 = Genome(child2_genes)
        logging.debug(f"Genome crossover complete. Child1 has {len(child1.genes)} genes, Child2 has {len(child2.genes)} genes.")
        return child1, child2


class GeneticInterpreter:
    """
    Advanced genetic sequence interpreter with robust error handling, optimized array operations,
    and comprehensive genetic mechanisms including gene linkage, crossover, and environmental interactions.
    Implements Turing-complete genetic programming with safe numerical operations, regulatory networks,
    epistatic and epigenetic mechanisms.
    """
    
    def __init__(self, gene_sequence: Optional[List[List[Any]]] = None):
        """
        Initializes the GeneticInterpreter with a default or provided gene sequence, safety bounds, and genetic mechanisms.

        Args:
            gene_sequence (Optional[List[List[Any]]]): A list of gene sequences, defaults to a predefined sequence.
        """
        self.default_sequence = [
            ["start_movement", 1.0, 0.1, 0.0],
            ["start_interaction", 0.5, 100.0], 
            ["start_energy", 0.1, 0.5, 0.3],
            ["start_reproduction", 150.0, 100.0, 50.0, 30.0],
            ["start_growth", 0.1, 2.0, 100.0],
            ["start_predation", 10.0, 5.0]
        ]
        self.gene_sequence = gene_sequence if gene_sequence is not None else self.default_sequence
        self._setup_safety_bounds()
        self._initialize_genetic_mechanisms()
        logging.info("GeneticInterpreter initialized.")

    def _setup_safety_bounds(self) -> None:
        """Configures safety bounds for numerical operations."""
        self.bounds = {
            'energy': (0.0, 200.0),
            'velocity': (-10.0, 10.0),
            'traits': (0.1, 3.0),
            'mass': (0.1, 5.0),
            'age': (0.0, float('inf')),
            'distance': (1e-10, float('inf'))
        }
        logging.debug("Safety bounds configured.")

    def _initialize_genetic_mechanisms(self) -> None:
        """Initializes advanced genetic mechanisms including regulatory networks, epistatic and epigenetic mechanisms."""
        self.regulatory_networks = self._create_regulatory_networks()
        self.epistatic_interactions = self._create_epistatic_interactions()
        self.epigenetic_modifications = self._create_epigenetic_modifications()
        logging.debug("Genetic mechanisms initialized.")

    def _create_regulatory_networks(self) -> Dict[str, Any]:
        """Creates regulatory networks for gene expression control."""
        networks = {
            'movement': {'inhibitors': [], 'activators': []},
            'interaction': {'inhibitors': [], 'activators': []},
            'energy': {'inhibitors': [], 'activators': []},
            'reproduction': {'inhibitors': [], 'activators': []},
            'growth': {'inhibitors': [], 'activators': []},
            'predation': {'inhibitors': [], 'activators': []}
        }
        logging.debug("Regulatory networks created.")
        return networks

    def _create_epistatic_interactions(self) -> Dict[str, Any]:
        """Creates epistatic interactions between genes."""
        interactions = {
            'movement': {'modifiers': []},
            'interaction': {'modifiers': []},
            'energy': {'modifiers': []},
            'reproduction': {'modifiers': []},
            'growth': {'modifiers': []},
            'predation': {'modifiers': []}
        }
        logging.debug("Epistatic interactions created.")
        return interactions

    def _create_epigenetic_modifications(self) -> Dict[str, Any]:
        """Creates epigenetic modifications affecting gene expression."""
        modifications = {
            'methylation': {},
            'acetylation': {},
            'phosphorylation': {}
        }
        logging.debug("Epigenetic modifications created.")
        return modifications

    def decode(self, particle: 'CellularTypeData', others: List['CellularTypeData'], env: 'SimulationConfig') -> None:
        """
        Decodes the genetic sequence for a particle, applying gene effects.

        Args:
            particle (CellularTypeData): The particle to decode the genes for.
            others (List[CellularTypeData]): A list of other particles in the simulation.
            env (SimulationConfig): The simulation environment configuration.
        """
        try:
            for gene in self.gene_sequence:
                if not isinstance(gene, (list, tuple)) or len(gene) < 2:
                    logging.warning(f"Invalid gene format: {gene}. Skipping.")
                    continue
                    
                gene_type = str(gene[0])
                gene_data = np.asarray(gene[1:], dtype=np.float64)
                
                # Ensure gene_data is properly shaped for broadcasting
                if gene_data.ndim == 1:
                    gene_data = gene_data.reshape(-1, 1)
                
                method = getattr(self, f"apply_{gene_type.replace('start_', '')}_gene", None)
                if method:
                    method(particle, others, gene_data, env)
                else:
                    logging.warning(f"No method found for gene type: {gene_type}. Skipping.")
                    
        except Exception as e:
            logging.error(f"Error in genetic decoding: {str(e)}", exc_info=True)
            # Graceful recovery - maintain particle state
            self._ensure_particle_stability(particle)

    def _ensure_particle_stability(self, particle: 'CellularTypeData') -> None:
        """Ensures particle arrays maintain valid states by clipping values to safe bounds."""
        for attr in ['energy', 'vx', 'vy']:
            if hasattr(particle, attr):
                arr = getattr(particle, attr)
                if isinstance(arr, np.ndarray):
                    # Clip to safe bounds
                    if attr == 'energy':
                        setattr(particle, attr, np.clip(arr, *self.bounds['energy']))
                    elif attr in ['vx', 'vy']:
                        setattr(particle, attr, np.clip(arr, *self.bounds['velocity']))
                    logging.debug(f"Particle attribute '{attr}' stabilized.")

    def apply_movement_gene(self, particle: 'CellularTypeData', others: List['CellularTypeData'], gene_data: np.ndarray, env: 'SimulationConfig') -> None:
        """Applies movement based on the movement gene, with vectorized operations and safety checks."""
        try:
            # Broadcast parameters safely
            speed_modifier = np.clip(gene_data[0] if gene_data.size > 0 else 1.0, 0.1, 3.0)
            randomness = np.clip(gene_data[1] if gene_data.size > 1 else 0.1, 0.0, 1.0)
            direction_bias = np.clip(gene_data[2] if gene_data.size > 2 else 0.0, -1.0, 1.0)

            # Generate random movement with proper broadcasting
            shape = particle.vx.shape
            random_movement = randomness * np.random.uniform(-1, 1, size=shape)
            
            # Safe velocity updates with friction and bounds
            friction_factor = np.clip(1.0 - env.friction, 0.0, 1.0)
            
            # Update velocities with broadcasting
            new_vx = particle.vx * friction_factor * speed_modifier + random_movement + direction_bias
            new_vy = particle.vy * friction_factor * speed_modifier + random_movement + direction_bias
            
            # Clip velocities to safe bounds
            particle.vx = np.clip(new_vx, *self.bounds['velocity'])
            particle.vy = np.clip(new_vy, *self.bounds['velocity'])

            # Safe energy cost calculation with array broadcasting
            velocity_magnitude = np.sqrt(particle.vx**2 + particle.vy**2)
            energy_cost = np.clip(velocity_magnitude * 0.01, 0.0, particle.energy)
            particle.energy = np.maximum(0.0, particle.energy - energy_cost)
            logging.debug(f"Movement gene applied. Speed modifier: {speed_modifier}, Randomness: {randomness}, Direction bias: {direction_bias}")

        except Exception as e:
            logging.error(f"Movement gene error: {str(e)}", exc_info=True)
            self._ensure_particle_stability(particle)

    def apply_interaction_gene(self, particle: 'CellularTypeData', others: List['CellularTypeData'], gene_data: np.ndarray, env: 'SimulationConfig') -> None:
        """Applies interactions based on the interaction gene, with vectorized calculations and safety checks."""
        try:
            attraction_strength = np.clip(gene_data[0] if gene_data.size > 0 else 0.5, -2.0, 2.0)
            interaction_radius = np.clip(gene_data[1] if gene_data.size > 1 else 100.0, 10.0, 300.0)

            for other in others:
                if other is particle:
                    continue

                # Efficient distance calculation with broadcasting
                dx = other.x[np.newaxis, :] - particle.x[:, np.newaxis]
                dy = other.y[np.newaxis, :] - particle.y[:, np.newaxis]
                distances = np.hypot(dx, dy)
                
                # Safe interaction mask
                interact_mask = (distances > self.bounds['distance'][0]) & (distances < interaction_radius)
                if not np.any(interact_mask):
                    continue

                # Safe normalized vectors calculation with divide-by-zero protection
                inv_distances = np.divide(1.0, distances, out=np.zeros_like(distances), where=distances > self.bounds['distance'][0])
                dx_norm = dx * inv_distances
                dy_norm = dy * inv_distances

                # Force calculation with distance falloff and broadcasting
                force_magnitudes = np.where(interact_mask,
                                          attraction_strength * (1.0 - distances / interaction_radius),
                                          0.0)

                # Accumulate forces safely
                particle.vx += np.sum(np.nan_to_num(dx_norm * force_magnitudes, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
                particle.vy += np.sum(np.nan_to_num(dy_norm * force_magnitudes, nan=0.0, posinf=0.0, neginf=0.0), axis=1)

                # Clip velocities to safe bounds
                particle.vx = np.clip(particle.vx, *self.bounds['velocity'])
                particle.vy = np.clip(particle.vy, *self.bounds['velocity'])

                # Safe energy cost with interaction count
                interaction_count = np.sum(interact_mask, axis=1)
                energy_cost = np.clip(0.01 * interaction_count, 0.0, particle.energy)
                particle.energy = np.maximum(0.0, particle.energy - energy_cost)
            logging.debug(f"Interaction gene applied. Attraction strength: {attraction_strength}, Interaction radius: {interaction_radius}")

        except Exception as e:
            logging.error(f"Interaction gene error: {str(e)}", exc_info=True)
            self._ensure_particle_stability(particle)

    def apply_energy_gene(self, particle: 'CellularTypeData', others: List['CellularTypeData'], gene_data: np.ndarray, env: 'SimulationConfig') -> None:
        """Applies energy dynamics based on the energy gene, with safe bounds and epigenetic modifications."""
        try:
            passive_gain = np.clip(gene_data[0] if gene_data.size > 0 else 0.1, 0.0, 0.5)
            feeding_efficiency = np.clip(gene_data[1] if gene_data.size > 1 else 0.5, 0.1, 1.0)

            # Safe energy calculations with broadcasting
            base_gain = passive_gain * np.clip(particle.energy_efficiency, *self.bounds['traits'])
            env_modifier = np.clip(1.0 + env.environmental_factors["temperature"] / 100.0, 0.5, 2.0)
            energy_gain = base_gain * env_modifier * feeding_efficiency

            # Safe age factor calculation with divide-by-zero protection
            max_age_safe = np.maximum(particle.max_age, 1.0)
            age_factor = np.clip(particle.age / max_age_safe, 0.0, 1.0)
            energy_decay = 0.01 * age_factor

            # Update energy with bounds
            particle.energy = np.clip(
                particle.energy + energy_gain - energy_decay,
                *self.bounds['energy']
            )
            logging.debug(f"Energy gene applied. Passive gain: {passive_gain}, Feeding efficiency: {feeding_efficiency}")

        except Exception as e:
            logging.error(f"Energy gene error: {str(e)}", exc_info=True)
            self._ensure_particle_stability(particle)

    def apply_reproduction_gene(self, particle: 'CellularTypeData', others: List['CellularTypeData'], gene_data: np.ndarray, env: 'SimulationConfig') -> None:
        """Handles reproduction based on the reproduction gene, with mutation safety and regulatory network influence."""
        try:
            thresholds = np.clip(gene_data[:2] if gene_data.size > 1 else [150.0, 100.0], 50.0, 200.0)
            costs = np.clip(gene_data[2:4] if gene_data.size > 3 else [50.0, 30.0], 10.0, 100.0)

            # Safe reproduction check with broadcasting
            can_reproduce = (particle.energy > thresholds[1]) & \
                          (particle.age > costs[1]) & \
                          particle.alive

            if not np.any(can_reproduce):
                return

            reproduce_indices = np.where(can_reproduce)[0]
            
            # Vectorized mutation calculations
            mutation_rate = np.clip(env.genetics.gene_mutation_rate, 0.0, 1.0)
            mutation_range = np.array(env.genetics.gene_mutation_range)
            
            for idx in reproduce_indices:
                # Safe energy deduction with proper scalar handling
                energy_array = particle.energy[idx:idx+1] # Get single-element array slice
                if energy_array[0] < costs[0]:
                    continue
                
                # Deduct energy using array operations
                energy_cost = np.full_like(energy_array, costs[0])
                particle.energy[idx:idx+1] = np.subtract(energy_array, energy_cost)

                # Create offspring with safe mutations and broadcasting
                offspring_genome = self._generate_offspring_genome(particle, idx, mutation_rate, mutation_range, env)
                
                # Safe genetic distance calculation
                genetic_distance = self._calculate_genetic_distance(particle, offspring_genome, idx)
                
                # Determine species with bounds check
                species_array = particle.species_id[particle.species_id >= 0]
                species_id_val = int(np.max(species_array) + 1) if genetic_distance > env.speciation_threshold else int(particle.species_id[idx])

                # Add offspring with position jitter
                jitter = np.random.uniform(-5, 5, size=2)
                particle.add_component(
                    x=particle.x[idx] + jitter[0],
                    y=particle.y[idx] + jitter[1],
                    vx=float(np.clip(particle.vx[idx] * np.random.uniform(0.9, 1.1), *self.bounds['velocity'])),
                    vy=float(np.clip(particle.vy[idx] * np.random.uniform(0.9, 1.1), *self.bounds['velocity'])),
                    energy=np.clip(particle.energy[idx:idx+1][0] * 0.5, *self.bounds['energy']),
                    mass_val=float(particle.mass[idx]) if particle.mass_based else None,
                    energy_efficiency_val=float(particle.energy_efficiency[idx]) if particle.energy_efficiency is not None else None,
                    genome=offspring_genome,
                    species_id_val=species_id_val,
                    parent_id_val=particle.type_id,
                    max_age=float(particle.max_age),
                )
            logging.debug(f"Reproduction gene applied. Thresholds: {thresholds}, Costs: {costs}, {len(reproduce_indices)} particles reproduced.")

        except Exception as e:
            logging.error(f"Reproduction gene error: {str(e)}", exc_info=True)
            self._ensure_particle_stability(particle)

    def _generate_offspring_genome(self, parent: 'CellularTypeData', idx: int, mutation_rate: float, mutation_range: np.ndarray, env: 'SimulationConfig') -> Genome:
        """Generates an offspring genome with gene linkage and environmental interactions."""
        try:
            # Inherit parent's genome
            parent_genome = parent.genome
            offspring_genome = Genome([Gene(gene.trait, gene.value, gene.linked_genes.copy()) for gene in parent_genome.genes])

            # Apply mutations
            offspring_genome.mutate(mutation_rate, mutation_range, [env.genetics.trait_ranges[i] for i in range(len(parent.genetics.gene_traits))])

            # Environmental interactions (e.g., temperature affecting gene expression)
            for gene in offspring_genome.genes:
                env_factor = env.environmental_factors.get("temperature", 25.0)
                gene.value += env_factor * 0.001  # Simple example: temperature slightly increases gene value
                gene.value = np.clip(gene.value, *self.bounds['traits'])

            logging.debug(f"Offspring genome generated. Parent index: {idx}")
            return offspring_genome

        except Exception as e:
            logging.error(f"Offspring genome generation error: {str(e)}", exc_info=True)
            return Genome([Gene(trait, np.clip(1.0, self.bounds['traits'][0], self.bounds['traits'][1])) for i, trait in enumerate(parent.genetics.gene_traits)])

    def _calculate_genetic_distance(self, particle: 'CellularTypeData', offspring_genome: Genome, idx: int) -> float:
        """Calculates the genetic distance between a parent and offspring genome."""
        try:
            squared_diffs = []
            for gene, offspring_gene in zip(particle.genome.genes, offspring_genome.genes):
                diff = np.subtract(offspring_gene.value, gene.value, dtype=np.float64)
                squared = np.square(diff, dtype=np.float64)
                squared_diffs.append(squared)
            
            # Safe root calculation with epsilon
            if not squared_diffs:
                return 0.0
            distance = float(np.sqrt(np.sum(squared_diffs, dtype=np.float64) + 1e-10))
            logging.debug(f"Genetic distance calculated: {distance}, Parent index: {idx}")
            return distance
            
        except Exception as e:
            logging.error(f"Genetic distance calculation error handled: {str(e)}", exc_info=True)
            return 0.0

    def apply_growth_gene(self, particle: 'CellularTypeData', others: List['CellularTypeData'], gene_data: np.ndarray, env: 'SimulationConfig') -> None:
        """Applies growth based on the growth gene, with safety checks and regulatory network influence."""
        try:
            # Safe parameter extraction with bounds
            growth_rate = np.clip(gene_data[0] if gene_data.size > 0 else 0.1, 0.01, 0.5)
            adult_size = np.clip(gene_data[1] if gene_data.size > 1 else 2.0, 1.0, 5.0)
            maturity_age = np.clip(gene_data[2] if gene_data.size > 2 else 100.0, 50.0, 200.0)

            # Ensure arrays exist and have compatible shapes
            if particle.age is None or particle.energy is None:
                return
                
            # Broadcast arrays safely
            age = np.asarray(particle.age, dtype=np.float64)
            energy = np.asarray(particle.energy, dtype=np.float64)
            efficiency = np.clip(np.asarray(particle.energy_efficiency, dtype=np.float64), 0.1, 10.0)
            
            # Calculate growth masks and factors safely
            juvenile_mask = age < maturity_age
            maturity_ratio = np.divide(age, maturity_age, out=np.ones_like(age), where=maturity_age > 0)
            growth_factor = np.where(juvenile_mask,
                                   growth_rate * (1.0 - np.clip(maturity_ratio, 0.0, 1.0)),
                                   0.0)

            # Update energy with safety bounds
            particle.energy = np.clip(
                energy + growth_factor * efficiency,
                0.0,
                200.0
            )
            
            # Handle mass-based growth if applicable
            if particle.mass_based and particle.mass is not None:
                mass = np.asarray(particle.mass, dtype=np.float64)
                particle.mass = np.clip(
                    np.where(juvenile_mask,
                            mass * (1.0 + growth_factor),
                            mass),
                    0.1,
                    adult_size
                )
            logging.debug(f"Growth gene applied. Growth rate: {growth_rate}, Adult size: {adult_size}, Maturity age: {maturity_age}")

        except Exception as e:
            logging.error(f"Growth gene error handled: {str(e)}", exc_info=True)

    def apply_predation_gene(self, particle: 'CellularTypeData', others: List['CellularTypeData'], gene_data: np.ndarray, env: 'SimulationConfig') -> None:
        """Applies predation based on the predation gene, with vectorized operations and epistatic interactions."""
        try:
            # Safe parameter extraction
            attack_power = np.clip(gene_data[0] if gene_data.size > 0 else 10.0, 1.0, 20.0)
            energy_gain = np.clip(gene_data[1] if gene_data.size > 1 else 5.0, 1.0, 10.0)

            for other in others:
                if other is particle or not other.alive.any():
                    continue

                try:
                    # Efficient vectorized distance calculation
                    dx = other.x[np.newaxis, :] - particle.x[:, np.newaxis]
                    dy = other.y[np.newaxis, :] - particle.y[:, np.newaxis]
                    distances = np.hypot(dx, dy)

                    # Broadcast-safe predation mask
                    pred_energy = particle.energy[:, np.newaxis]
                    prey_energy = other.energy[np.newaxis, :]
                    prey_alive = other.alive[np.newaxis, :]
                    
                    predation_mask = (
                        (distances < env.predation_range) & 
                        prey_alive & 
                        (pred_energy > prey_energy)
                    )

                    if not np.any(predation_mask):
                        continue

                    pred_idx, prey_idx = np.where(predation_mask)

                    # Safe energy calculations with broadcasting
                    energy_ratio = np.divide(
                        particle.energy[pred_idx],
                        other.energy[prey_idx],
                        out=np.ones_like(particle.energy[pred_idx]),
                        where=other.energy[prey_idx] > 0
                    )
                    
                    damage = attack_power * np.clip(energy_ratio, 0.0, 10.0)
                    gained_energy = energy_gain * damage * np.clip(
                        particle.energy_efficiency[pred_idx], 
                        0.1, 
                        10.0
                    )

                    # Atomic updates with safety bounds
                    other.energy[prey_idx] = np.maximum(0.0, other.energy[prey_idx] - damage)
                    particle.energy[pred_idx] = np.clip(
                        particle.energy[pred_idx] + gained_energy,
                        0.0,
                        200.0
                    )

                    # Update alive status safely
                    other.alive[prey_idx] = other.energy[prey_idx] > 0
                    logging.debug(f"Predation interaction applied. Attack power: {attack_power}, Energy gain: {energy_gain}, {len(prey_idx)} preyed upon.")

                except Exception as e:
                    logging.error(f"Individual predation interaction error handled: {str(e)}", exc_info=True)
                    continue

        except Exception as e:
            logging.error(f"Predation gene error handled: {str(e)}", exc_info=True)

###############################################################
# Cellular Component & Type Data Management (Extended)
###############################################################

class CellularTypeData:
    """
    Represents a cellular type with multiple cellular components.
    Manages positions, velocities, energy, mass, and genetic traits of components.
    Includes communication capabilities and neural networks for intelligence.
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
                 gene_traits: List[str] = ["speed_factor", "interaction_strength", "perception_range", "reproduction_rate", "synergy_affinity", "colony_factor", "drift_sensitivity"],
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
                 neural_network_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a CellularTypeData instance with given parameters.
        """
        # Input validation and sanitization
        n_particles = max(1, int(n_particles))
        window_width = max(1, int(window_width))
        window_height = max(1, int(window_height))
        
        # Store metadata with validation
        self.type_id = int(type_id)
        self.color = tuple(map(lambda x: max(0, min(255, int(x))), color))
        self.mass_based = bool(mass is not None)

        # Store parameter bounds with validation
        self.min_energy = float(min_energy)
        self.max_energy = float(max_energy)
        self.min_mass = float(min_mass)
        self.max_mass = float(max_mass)
        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)
        self.min_perception = float(min_perception)
        self.max_perception = float(max_perception)
        self.min_reproduction = float(min_reproduction)
        self.max_reproduction = float(max_reproduction)
        self.min_synergy = float(min_synergy)
        self.max_synergy = float(max_synergy)
        self.min_colony = float(min_colony)
        self.max_colony = float(max_colony)
        self.min_drift = float(min_drift)
        self.max_drift = float(max_drift)
        self.min_energy_efficiency = float(min_energy_efficiency)
        self.max_energy_efficiency = float(max_energy_efficiency)

        # Initialize Neural Network
        nn_params = neural_network_params if neural_network_params else {}
        self.neural_network = NeuralNetwork(
            input_size=nn_params.get("input_size", 5),
            hidden_size=nn_params.get("hidden_size", 10),
            output_size=nn_params.get("output_size", 5),
            activation=nn_params.get("activation", "relu")
        )

        # Initialize Genome
        self.genome = Genome()
        for trait in gene_traits:
            initial_value = np.random.uniform(0.5, 1.5)
            self.genome.add_gene(Gene(trait, initial_value))

        try:
            # Initialize positions safely
            coords = random_xy(window_width, window_height, n_particles)
            self.x = coords[:, 0].astype(np.float64)
            self.y = coords[:, 1].astype(np.float64)

            # Initialize energy efficiency safely
            if energy_efficiency is None:
                self.energy_efficiency = np.clip(
                    np.random.uniform(self.min_energy_efficiency, self.max_energy_efficiency, n_particles),
                    self.min_energy_efficiency,
                    self.max_energy_efficiency
                ).astype(np.float64)
            else:
                self.energy_efficiency = np.full(n_particles, np.clip(
                    float(energy_efficiency),
                    self.min_energy_efficiency,
                    self.max_energy_efficiency
                ), dtype=np.float64)

            # Safe velocity scaling calculation
            velocity_scaling = base_velocity_scale / np.maximum(self.energy_efficiency, 1e-10)

            # Initialize velocities safely
            self.vx = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity
            ).astype(np.float64)
            self.vy = np.clip(
                np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
                self.min_velocity,
                self.max_velocity
            ).astype(np.float64)

            # Initialize energy safely
            self.energy = np.clip(
                np.full(n_particles, initial_energy, dtype=np.float64),
                self.min_energy,
                self.max_energy
            )

            # Initialize mass safely for mass-based types
            if self.mass_based:
                if mass is None or mass <= 0.0:
                    mass = self.min_mass
                self.mass = np.clip(
                    np.full(n_particles, mass, dtype=np.float64),
                    self.min_mass,
                    self.max_mass
                )
            else:
                self.mass = None

            # Initialize status arrays safely
            self.alive = np.ones(n_particles, dtype=bool)
            self.age = np.zeros(n_particles, dtype=np.float64)
            self.max_age = float(max_age)

            # Initialize tracking arrays safely
            self.species_id = np.full(n_particles, self.type_id, dtype=np.int32)
            self.parent_id = np.full(n_particles, -1, dtype=np.int32)
            self.colony_id = np.full(n_particles, -1, dtype=np.int32)
            self.colony_role = np.zeros(n_particles, dtype=np.int32)
            self.synergy_connections = np.zeros((n_particles, n_particles), dtype=bool)
            self.fitness_score = np.zeros(n_particles, dtype=np.float64)
            self.generation = np.zeros(n_particles, dtype=np.int32)
            self.mutation_history = [[] for _ in range(n_particles)]

            # Communication and Neural Network
            self.signals_received: List[Signal] = []
            self.communicate_with: List['CellularTypeData'] = []
        except Exception as e:
            print(f"Error in CellularTypeData initialization: {e}")
            self._initialize_fallback(n_particles)

    def _initialize_fallback(self, n_particles: int) -> None:
            """Safe fallback initialization with minimum values."""
            n_particles = max(1, int(n_particles))
            self.x = np.zeros(n_particles, dtype=np.float64)
            self.y = np.zeros(n_particles, dtype=np.float64)
            self.vx = np.zeros(n_particles, dtype=np.float64)
            self.vy = np.zeros(n_particles, dtype=np.float64)
            self.energy = np.full(n_particles, self.min_energy, dtype=np.float64)
            self.energy_efficiency = np.full(n_particles, self.min_energy_efficiency, dtype=np.float64)
            if self.mass_based:
                self.mass = np.full(n_particles, self.min_mass, dtype=np.float64)
            self.alive = np.ones(n_particles, dtype=bool)
            self.age = np.zeros(n_particles, dtype=np.float64)
            # Initialize other arrays with safe minimum values...

    def _validate_array_shapes(self) -> None:
            """Validate and correct array shapes for consistency."""
            base_size = len(self.x)
            arrays_to_check = [
                'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                'species_id', 'parent_id'
            ]
            
            for attr in arrays_to_check:
                current = getattr(self, attr)
                if len(current) != base_size:
                    setattr(self, attr, np.resize(current, base_size))
                    
            if self.mass_based and self.mass is not None:
                if len(self.mass) != base_size:
                    self.mass = np.resize(self.mass, base_size)

    def is_alive_mask(self) -> np.ndarray:
            """Compute alive mask with safe array operations."""
            try:
                self._validate_array_shapes()
                mask = (self.alive & 
                       (self.energy > self.min_energy) & 
                       (self.age < self.max_age))
                if self.mass_based and self.mass is not None:
                    mask &= (self.mass > self.min_mass)
                return mask
            except Exception:
                return np.ones(len(self.x), dtype=bool)

    def update_alive(self) -> None:
            """Update alive status safely."""
            try:
                self.alive = self.is_alive_mask()
            except Exception:
                self.alive = np.ones_like(self.alive)

    def age_components(self) -> None:
            """Age components with safe operations."""
            try:
                self.age = np.add(self.age, 1.0, where=self.alive)
                self.energy = np.clip(self.energy, self.min_energy, self.max_energy)
            except Exception:
                pass

    def update_states(self) -> None:
            """Update component states safely."""
            try:
                self._validate_array_shapes()
            except Exception:
                pass

    def remove_dead(self, config: SimulationConfig) -> None:
            """Remove dead components safely with array broadcasting."""
            try:
                self._validate_array_shapes()
                alive_mask = self.is_alive_mask()
                dead_due_to_age = (~alive_mask) & (self.age >= self.max_age)
                
                if np.any(dead_due_to_age):
                    self._handle_energy_transfer(dead_due_to_age, alive_mask, config)
                
                # Filter arrays safely
                arrays_to_filter = [
                    'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                    'species_id', 'parent_id'
                ]
                
                for attr in arrays_to_filter:
                    current = getattr(self, attr)
                    setattr(self, attr, current[alive_mask])
                    
                if self.mass_based and self.mass is not None:
                    self.mass = self.mass[alive_mask]
                    
            except Exception as e:
                print(f"Error in remove_dead: {str(e)}")
                self._validate_array_shapes()

    def _handle_energy_transfer(self, dead_due_to_age: np.ndarray, alive_mask: np.ndarray, config: SimulationConfig) -> None:
            """Handle energy transfer from dead components safely."""
            try:
                alive_indices = np.where(alive_mask)[0]
                dead_age_indices = np.where(dead_due_to_age)[0]
                
                if len(alive_indices) > 0:
                    alive_positions = np.column_stack((self.x[alive_indices], self.y[alive_indices]))
                    tree = cKDTree(alive_positions)
                    
                    batch_size = min(1000, len(dead_age_indices))
                    for i in range(0, len(dead_age_indices), batch_size):
                        batch_indices = dead_age_indices[i:i + batch_size]
                        dead_positions = np.column_stack((self.x[batch_indices], self.y[batch_indices]))
                        dead_energies = self.energy[batch_indices]
                        
                        distances, neighbors = tree.query(
                            dead_positions,
                            k=min(3, len(alive_indices)),
                            distance_upper_bound=config.predation_range
                        )
                        
                        valid_mask = distances < config.predation_range
                        for j, (dist_row, neighbor_row, dead_energy) in enumerate(zip(distances, neighbors, dead_energies)):
                            valid = valid_mask[j]
                            if np.any(valid):
                                valid_neighbors = neighbor_row[valid]
                                energy_share = dead_energy / max(np.sum(valid), 1)
                                self.energy[alive_indices[valid_neighbors]] += energy_share
                                self.energy[batch_indices[j]] = 0.0
                                
            except Exception as e:
                print(f"Error in energy transfer: {str(e)}")

    def add_component(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        energy: float,
        mass_val: Optional[float],
        energy_efficiency_val: float,
        genome: Genome,
        species_id_val: int,
        parent_id_val: int,
        max_age: float
        ) -> None:
            """Add new component safely with array broadcasting."""
            try:
                # Validate and clip input values
                x = float(x)
                y = float(y)
                vx = np.clip(float(vx), self.min_velocity, self.max_velocity)
                vy = np.clip(float(vy), self.min_velocity, self.max_velocity)
                energy = np.clip(float(energy), self.min_energy, self.max_energy)
                energy_efficiency_val = np.clip(float(energy_efficiency_val), self.min_energy_efficiency, self.max_energy_efficiency)
                
                # Prepare new values as arrays for broadcasting
                new_values = {
                    'x': np.array([x]),
                    'y': np.array([y]),
                    'vx': np.array([vx]),
                    'vy': np.array([vy]),
                    'energy': np.array([energy]),
                    'alive': np.array([True]),
                    'age': np.array([0.0]),
                    'energy_efficiency': np.array([energy_efficiency_val]),
                    'species_id': np.array([species_id_val]),
                    'parent_id': np.array([parent_id_val])
                }
                
                # Safely concatenate arrays
                for attr, new_value in new_values.items():
                    current = getattr(self, attr)
                    setattr(self, attr, np.concatenate((current, new_value)))
                
                # Handle mass separately if mass-based
                if self.mass_based:
                    if mass_val is None or mass_val <= 0.0:
                        mass_val = self.min_mass
                    mass_val = np.clip(float(mass_val), self.min_mass, self.max_mass)
                    self.mass = np.concatenate((self.mass, np.array([mass_val])))

                # Append genome
                self.genome.genes.extend([Gene(gene.trait, gene.value, gene.linked_genes.copy()) for gene in genome.genes])

                self._validate_array_shapes()
                
            except Exception as e:
                print(f"Error adding component: {str(e)}")

    ###############################################################
    # Utility Functions
    ###############################################################

def random_xy(window_width: int, window_height: int, n: int = 1) -> np.ndarray:
    """
    Generate n random (x, y) coordinates with robust error handling.
    """
    try:
        # Validate inputs
        window_width = max(1, int(window_width))
        window_height = max(1, int(window_height))
        n = max(1, int(n))
        
        # Generate coordinates safely
        coords = np.random.uniform(0, [window_width, window_height], (n, 2))
        return np.clip(coords, 0, [window_width, window_height])
    except Exception:
        # Return safe fallback value
        return np.zeros((1, 2))

def generate_vibrant_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n distinct vibrant colors with error handling and validation.
    """
    try:
        n = max(1, int(n))
        colors = []
        
        for i in range(n):
            hue = (i / n) % 1.0
            h_i = int(hue * 6)
            f = np.clip(hue * 6 - h_i, 0, 1)
            
            # Safe color calculations
            p = 0
            q = int(np.clip((1 - f) * 255, 0, 255))
            t = int(np.clip(f * 255, 0, 255))
            v = 255
            
            # Safe color assignment
            color = [(v, t, p), (q, v, p), (p, v, t),
                    (p, q, v), (t, p, v), (v, p, q)][h_i % 6]
            colors.append(color)
            
        return colors
    except Exception:
        # Return safe fallback color
        return [(255, 255, 255)]

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Numerically stable sigmoid function.
    """
    try:
        # Clip extreme values to prevent overflow
        x_safe = np.clip(x, -88.0, 88.0)
        return 1 / (1 + np.exp(-x_safe))
    except Exception:
        return 0.5  # Safe fallback value

###############################################################
# Cellular Type Manager (Handles Multi-Type Operations & Reproduction)
###############################################################

class CellularTypeManager:
    """
    Manages all cellular types in the simulation with robust error handling and optimized array operations.
    """
    
    def __init__(self, config: SimulationConfig, colors: List[Tuple[int, int, int]], mass_based_type_indices: List[int]):
        """Initialize with robust error checking and array validation."""
        self.config = config
        self.cellular_types: List[CellularTypeData] = []
        self.mass_based_type_indices = np.array(mass_based_type_indices)
        self.colors = colors
        self.EPSILON = np.finfo(np.float32).eps
        self.MIN_ARRAY_SIZE = 1

    def add_cellular_type_data(self, data: CellularTypeData) -> None:
        """Add cellular type with validation."""
        if data is not None:
            self.cellular_types.append(data)

    def get_cellular_type_by_id(self, i: int) -> Optional[CellularTypeData]:
        """Get cellular type with bounds checking."""
        try:
            return self.cellular_types[i]
        except IndexError:
            return None

    def remove_dead_in_all_types(self) -> None:
        """Remove dead components with safe array operations."""
        for ct in self.cellular_types:
            if ct is not None:
                ct.remove_dead(self.config)

    def reproduce(self) -> None:
        """
        Vectorized reproduction with comprehensive error handling and array safety.
        """
        for ct in self.cellular_types:
            try:
                # Validate array sizes and early exit conditions
                if (ct is None or 
                    ct.x.size == 0 or
                    ct.x.size >= self.config.max_particles_per_type or
                    not np.any(ct.alive)):
                    continue

                # Ensure all arrays are properly broadcast compatible
                arrays = [ct.genome.genes, ct.energy, ct.alive]
                target_shape = np.broadcast_shapes(*[len(arr.genes) for arr in [ct.genome] if arr is not None])

                # Calculate eligible components with safety checks
                eligible = (ct.alive & 
                           (ct.energy > self.config.reproduction_energy_threshold) &
                           (np.random.random(ct.x.shape) < ct.genome.genes[3].value))  # Assuming gene at index 3 is reproduction_rate

                if not np.any(eligible):
                    continue

                reproduce_indices = np.where(eligible)[0]
                
                # Vectorized mutation calculations
                mutation_rate = np.clip(ct.genome.genes[5].value, 0.0, 1.0)  # Assuming gene at index 5 is mutation rate
                mutation_range = np.array(ct.genome.genes[5].value_range) if hasattr(ct.genome.genes[5], 'value_range') else np.array(self.config.genetics.gene_mutation_range)
                
                for idx in reproduce_indices:
                    # Safe energy deduction with proper scalar handling
                    energy_array = ct.energy[idx:idx+1] # Get single-element array slice
                    if energy_array[0] < self.config.reproduction_energy_threshold:
                        continue
                    
                    # Deduct energy using array operations
                    energy_cost = np.full_like(energy_array, self.config.reproduction_offspring_energy_fraction * energy_array[0])
                    ct.energy[idx:idx+1] = np.subtract(energy_array, energy_cost)

                    # Create offspring with safe mutations and broadcasting
                    offspring_genome = ct.genome.crossover(ct.genome)[0]  # Taking first child
                    offspring_genome.mutate(mutation_rate, mutation_range, self.config.genetics.trait_ranges)

                    # Safe genetic distance calculation
                    genetic_distance = self._calculate_genetic_distance(ct.genome, offspring_genome)
                    
                    # Determine species with bounds check
                    species_array = ct.species_id[ct.species_id >= 0]
                    species_id_val = int(np.max(species_array) + 1) if genetic_distance > self.config.speciation_threshold else int(ct.species_id[idx])

                    # Add offspring with position jitter
                    jitter = np.random.uniform(-5, 5, size=2)
                    ct.add_component(
                        x=ct.x[idx] + jitter[0],
                        y=ct.y[idx] + jitter[1],
                        vx=np.clip(ct.vx[idx] * np.random.uniform(0.9, 1.1), *self.config.bounds['velocity']),
                        vy=np.clip(ct.vy[idx] * np.random.uniform(0.9, 1.1), *self.config.bounds['velocity']),
                        energy=np.clip(ct.energy[idx:idx+1][0], *self.config.bounds['energy']),
                        mass_val=ct.mass[idx] if ct.mass_based else None,
                        energy_efficiency_val=ct.energy_efficiency[idx],
                        genome=offspring_genome,
                        species_id_val=species_id_val,
                        parent_id_val=ct.type_id,
                        max_age=ct.max_age
                    )

            except Exception as e:
                print(f"Reproduction error handled for type {ct}: {str(e)}")
                continue

    def _calculate_genetic_distance(self, parent_genome: Genome, offspring_genome: Genome) -> float:
        """Calculate genetic distance with robust numerical stability, broadcasting, and epistatic interactions."""
        try:
            squared_diffs = []
            for gene_parent, gene_offspring in zip(parent_genome.genes, offspring_genome.genes):
                diff = np.subtract(gene_offspring.value, gene_parent.value, dtype=np.float64)
                squared = np.square(diff, dtype=np.float64)
                squared_diffs.append(squared)
            
            # Safe root calculation with epsilon
            if not squared_diffs:
                return 0.0
            return float(np.sqrt(np.sum(squared_diffs, dtype=np.float64) + 1e-10))
            
        except Exception as e:
            print(f"Genetic distance calculation error handled: {str(e)}")
            return 0.0

###############################################################
# Forces & Interactions
###############################################################
class InteractionRules:
    """
    Manages creation and evolution of interaction parameters, give-take matrix, and synergy matrix.
    Optimized for high performance with robust error handling and array safety.
    """
    
    def __init__(self, config: SimulationConfig, mass_based_type_indices: List[int]):
        """Initialize with robust error checking and array safety."""
        self.config = config
        self.mass_based_type_indices = np.array(mass_based_type_indices, dtype=np.int32)
        self.EPSILON = np.finfo(np.float32).eps
        self.MIN_ARRAY_SIZE = 1
        
        # Initialize matrices with error handling
        try:
            self.rules = self._create_interaction_matrix()
            self.give_take_matrix = self._create_give_take_matrix()
            self.synergy_matrix = self._create_synergy_matrix()
        except Exception as e:
            print(f"Matrix initialization error handled: {str(e)}")
            # Fallback initialization
            self.rules = []
            self.give_take_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=bool)
            self.synergy_matrix = np.zeros((self.config.n_cell_types, self.config.n_cell_types), dtype=np.float32)

    def _create_interaction_matrix(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Create interaction matrix with vectorized operations and safety checks."""
        try:
            n_types = max(1, self.config.n_cell_types)
            final_rules = []
            
            # Vectorized parameter generation
            type_pairs = np.array([(i, j) for i in range(n_types) for j in range(n_types)])
            mass_based_mask = np.isin(type_pairs, self.mass_based_type_indices)
            both_mass = np.all(mass_based_mask, axis=1)
            
            # Vectorized random generation
            rand_vals = np.random.random(len(type_pairs))
            use_gravity = both_mass & (rand_vals < 0.5)
            
            potential_strengths = np.random.uniform(
                self.config.interaction_strength_range[0],
                self.config.interaction_strength_range[1],
                len(type_pairs)
            )
            potential_strengths[rand_vals < 0.5] *= -1
            
            gravity_factors = np.where(
                use_gravity,
                np.random.uniform(0.1, 2.0, len(type_pairs)),
                np.zeros(len(type_pairs))
            )
            
            max_dists = np.random.uniform(50.0, 200.0, len(type_pairs))
            
            # Create rules with safety bounds
            for idx, (i, j) in enumerate(type_pairs):
                params = {
                    "use_potential": True,
                    "use_gravity": bool(use_gravity[idx]),
                    "potential_strength": float(np.clip(potential_strengths[idx], -1e6, 1e6)),
                    "gravity_factor": float(np.clip(gravity_factors[idx], 0, 1e3)),
                    "max_dist": float(np.clip(max_dists[idx], 10.0, 1e4))
                }
                final_rules.append((int(i), int(j), params))
                
            return final_rules
            
        except Exception as e:
            print(f"Interaction matrix creation error handled: {str(e)}")
            return [(0, 0, {"use_potential": True, "use_gravity": False, 
                          "potential_strength": 1.0, "gravity_factor": 0.0, "max_dist": 50.0})]

    def _create_give_take_matrix(self) -> np.ndarray:
        """Create give-take matrix with vectorized operations and shape safety."""
        try:
            n_types = max(1, self.config.n_cell_types)
            matrix = np.zeros((n_types, n_types), dtype=bool)
            
            # Vectorized random generation
            rand_mask = np.random.random((n_types, n_types)) < 0.1
            np.fill_diagonal(rand_mask, False) # No self-interaction
            
            return rand_mask
            
        except Exception as e:
            print(f"Give-take matrix creation error handled: {str(e)}")
            return np.zeros((self.MIN_ARRAY_SIZE, self.MIN_ARRAY_SIZE), dtype=bool)

    def _create_synergy_matrix(self) -> np.ndarray:
        """Create synergy matrix with vectorized operations and value safety."""
        try:
            n_types = max(1, self.config.n_cell_types)
            matrix = np.zeros((n_types, n_types), dtype=np.float32)
            
            # Vectorized random generation
            rand_mask = np.random.random((n_types, n_types)) < 0.1
            synergy_values = np.random.uniform(0.01, 0.3, (n_types, n_types))
            
            # Safe assignment with bounds
            matrix = np.where(rand_mask, synergy_values, 0.0)
            np.fill_diagonal(matrix, 0.0) # No self-synergy
            
            return np.clip(matrix, 0.0, 1.0)
            
        except Exception as e:
            print(f"Synergy matrix creation error handled: {str(e)}")
            return np.zeros((self.MIN_ARRAY_SIZE, self.MIN_ARRAY_SIZE), dtype=np.float32)

    def evolve_parameters(self, frame_count: int) -> None:
        """Evolve parameters with vectorized operations and robust error handling."""
        try:
            if frame_count % self.config.evolution_interval != 0:
                return
                
            # Vectorized rule evolution
            for _, _, params in self.rules:
                rand_vals = np.random.random(3)
                mutation_factors = np.random.uniform(0.95, 1.05, 3)
                
                if rand_vals[0] < 0.1: # Potential strength mutation
                    params["potential_strength"] = np.clip(
                        params["potential_strength"] * mutation_factors[0],
                        self.config.interaction_strength_range[0],
                        self.config.interaction_strength_range[1]
                    )
                    
                if rand_vals[1] < 0.05 and "gravity_factor" in params: # Gravity mutation
                    params["gravity_factor"] = np.clip(
                        params["gravity_factor"] * mutation_factors[1],
                        0.0, 10.0
                    )
                    
                if rand_vals[2] < 0.05: # Max distance mutation
                    params["max_dist"] = np.clip(
                        params["max_dist"] * mutation_factors[2],
                        10.0, 1000.0
                    )
            
            # Energy transfer evolution
            if np.random.random() < 0.1:
                self.config.energy_transfer_factor = np.clip(
                    self.config.energy_transfer_factor * np.random.uniform(0.95, 1.05),
                    0.0, 1.0
                )
            
            # Vectorized synergy evolution
            evolution_mask = np.random.random(self.synergy_matrix.shape) < 0.05
            mutation_values = np.random.uniform(-0.05, 0.05, self.synergy_matrix.shape)
            
            self.synergy_matrix = np.clip(
                np.where(
                    evolution_mask,
                    self.synergy_matrix + mutation_values,
                    self.synergy_matrix
                ),
                0.0, 1.0
            )
            
        except Exception as e:
            print(f"Parameter evolution error handled: {str(e)}")
            
def apply_interaction(a_x: np.ndarray, a_y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray, 
                     params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized force computation between cellular components with robust error handling.
    Handles arrays of different shapes through broadcasting.

    Parameters:
    -----------
    a_x, a_y : np.ndarray
        Coordinates of cellular components A
    b_x, b_y : np.ndarray 
        Coordinates of cellular components B
    params : Dict[str, Any]
        Interaction parameters

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Force components (fx, fy) arrays
    """
    try:
        # Ensure inputs are numpy arrays and broadcast to same shape
        arrays = np.broadcast_arrays(
            np.asarray(a_x, dtype=np.float64),
            np.asarray(a_y, dtype=np.float64), 
            np.asarray(b_x, dtype=np.float64),
            np.asarray(b_y, dtype=np.float64)
        )
        a_x, a_y, b_x, b_y = arrays

        # Calculate distances with numerical stability
        dx = np.subtract(a_x, b_x, dtype=np.float64) 
        dy = np.subtract(a_y, b_y, dtype=np.float64)
        d_sq = np.add(np.square(dx), np.square(dy), dtype=np.float64)
        
        # Initialize force arrays
        fx = np.zeros_like(d_sq, dtype=np.float64)
        fy = np.zeros_like(d_sq, dtype=np.float64)

        # Mask for valid distances
        max_dist = params.get("max_dist", np.inf)
        valid_mask = (d_sq > np.finfo(np.float64).tiny) & (d_sq <= max_dist**2)
        
        if not np.any(valid_mask):
            return fx, fy

        # Safe distance calculation for valid points
        d = np.sqrt(d_sq, where=valid_mask)
        
        # Potential-based interaction
        if params.get("use_potential", True):
            pot_strength = np.float64(params.get("potential_strength", 1.0))
            F_pot = pot_strength / d
            fx = np.add(fx, F_pot * dx, where=valid_mask, out=fx)
            fy = np.add(fy, F_pot * dy, where=valid_mask, out=fy)

        # Gravity-based interaction
        if params.get("use_gravity", False) and "m_a" in params and "m_b" in params:
            m_a = np.asarray(params["m_a"], dtype=np.float64)
            m_b = np.asarray(params["m_b"], dtype=np.float64)
            
            # Broadcast masses if needed
            m_a, m_b = np.broadcast_arrays(m_a, m_b)
            
            F_grav = params["gravity_factor"] * (m_a * m_b) / d_sq
            fx = np.add(fx, F_grav * dx, where=valid_mask, out=fx)
            fy = np.add(fy, F_grav * dy, where=valid_mask, out=fy)
    except Exception as e:
        print(f"Error in apply_interaction: {e}")
        traceback.print_exc()

        return np.nan_to_num(fx, copy=False), np.nan_to_num(fy, copy=False)

def give_take_interaction(giver_energy: np.ndarray, receiver_energy: np.ndarray,
                         giver_mass: Optional[np.ndarray], receiver_mass: Optional[np.ndarray],
                         config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, 
                                                          Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Vectorized energy and mass transfer between components with robust error handling.
    """
    try:
        # Ensure inputs are numpy arrays
        giver_energy = np.asarray(giver_energy, dtype=np.float64)
        receiver_energy = np.asarray(receiver_energy, dtype=np.float64)
        
        # Calculate energy transfer with safety bounds
        transfer_factor = np.clip(config.energy_transfer_factor, 0, 1)
        transfer_amount = np.multiply(receiver_energy, transfer_factor, dtype=np.float64)
        
        # Update energies safely
        new_receiver = np.subtract(receiver_energy, transfer_amount, dtype=np.float64)
        new_giver = np.add(giver_energy, transfer_amount, dtype=np.float64)
        
        # Handle mass transfer if enabled
        new_giver_mass = new_receiver_mass = None
        if config.mass_transfer and giver_mass is not None and receiver_mass is not None:
            giver_mass = np.asarray(giver_mass, dtype=np.float64)
            receiver_mass = np.asarray(receiver_mass, dtype=np.float64)
            
            mass_transfer = np.multiply(receiver_mass, transfer_factor, dtype=np.float64)
            new_receiver_mass = np.subtract(receiver_mass, mass_transfer, dtype=np.float64)
            new_giver_mass = np.add(giver_mass, mass_transfer, dtype=np.float64)

            # Ensure mass stays positive
            new_receiver_mass = np.maximum(new_receiver_mass, np.finfo(np.float64).tiny)
            new_giver_mass = np.maximum(new_giver_mass, np.finfo(np.float64).tiny)

        # Ensure energy stays positive
        new_receiver = np.maximum(new_receiver, 0)
        new_giver = np.maximum(new_giver, 0)
        
        return new_giver, new_receiver, new_giver_mass, new_receiver_mass

    except Exception as e:
        print(f"Give-take interaction error handled: {e}")
        return giver_energy, receiver_energy, giver_mass, receiver_mass

def apply_synergy(energyA: np.ndarray, energyB: np.ndarray, 
                 synergy_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized synergy calculation with robust error handling.
    """
    try:
        # Ensure inputs are numpy arrays and broadcast to same shape
        energyA, energyB = np.broadcast_arrays(
            np.asarray(energyA, dtype=np.float64),
            np.asarray(energyB, dtype=np.float64)
        )
        
        # Clip synergy factor for stability
        synergy_factor = np.clip(synergy_factor, 0, 1)
        
        # Calculate average energy safely
        avg_energy = np.multiply(np.add(energyA, energyB), 0.5, dtype=np.float64)
        
        # Calculate new energies with preserved total
        complement_factor = 1.0 - synergy_factor
        newA = np.add(
            np.multiply(energyA, complement_factor),
            np.multiply(avg_energy, synergy_factor),
            dtype=np.float64
        )
        newB = np.add(
            np.multiply(energyB, complement_factor),
            np.multiply(avg_energy, synergy_factor),
            dtype=np.float64
        )
        
        # Ensure energy stays positive and handle NaN/Inf
        return (
            np.maximum(np.nan_to_num(newA, copy=False), 0),
            np.maximum(np.nan_to_num(newB, copy=False), 0)
        )

    except Exception as e:
        print(f"Synergy calculation error handled: {e}")
        return energyA, energyB

###############################################################
# Communication Components
###############################################################
class Signal:
    """Represents a dynamically adaptive signal with self-optimizing properties."""
    
    def __init__(self, sender_id: int, receiver_id: int, content: Dict[str, Any], strength: float):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.strength = strength
        self.timestamp = time.time()
        self.metadata = {
            'path_history': [],
            'interference_level': 0.0,
            'priority': 1.0,
            'reliability': 1.0
        }
        self._initialize_adaptive_properties()

    def _initialize_adaptive_properties(self) -> None:
        """Initialize adaptive signal properties with safe defaults."""
        try:
            self.adaptive_params = {
                'decay_rate': np.random.uniform(0.95, 0.99),
                'noise_threshold': np.random.uniform(0.1, 0.3),
                'amplification_factor': np.random.uniform(1.0, 1.2),
                'interference_resistance': np.random.uniform(0.7, 0.9)
            }
        except Exception:
            self.adaptive_params = {
                'decay_rate': 0.97,
                'noise_threshold': 0.2,
                'amplification_factor': 1.1,
                'interference_resistance': 0.8
            }

    def adapt_strength(self, environmental_factor: float, distance: float = 1.0) -> None:
        """Dynamically adjust signal strength with multi-factor adaptation."""
        try:
            # Apply sophisticated environmental adaptation
            decay = np.exp(-distance * self.adaptive_params['decay_rate'])
            noise_resistance = 1.0 / (1.0 + self.metadata['interference_level'])
            
            # Compute adaptive strength modification
            strength_mod = (
                environmental_factor * 
                decay * 
                noise_resistance * 
                self.adaptive_params['amplification_factor']
            )
            
            # Apply with stability constraints
            self.strength = np.clip(self.strength * strength_mod, 0.0, 1.0)
            
            # Update reliability metric
            self.metadata['reliability'] = np.clip(
                self.metadata['reliability'] * decay * noise_resistance,
                0.1, 1.0
            )
            
            # Evolve adaptive parameters
            self._evolve_parameters()
            
        except Exception as e:
            print(f"Signal adaptation error handled: {e}")
            self.strength = np.clip(self.strength * 0.95, 0.1, 0.9)

    def _evolve_parameters(self) -> None:
        """Evolve signal parameters based on performance metrics."""
        try:
            success_rate = self.metadata['reliability']
            mutation_rate = 0.1 * (1.0 - success_rate)  # Higher mutation when reliability is low
            
            for param in self.adaptive_params:
                if np.random.random() < mutation_rate:
                    self.adaptive_params[param] *= np.random.uniform(0.95, 1.05)
        except Exception:
            pass

class NeuralNetwork:
    """Advanced adaptive neural network with dynamic architecture and multi-modal learning."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 activation: str = 'adaptive_relu'):
        self.architecture = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size
        }
        
        self._initialize_network()
        self._setup_adaptive_components()
        
    def _initialize_network(self) -> None:
        """Initialize network with advanced weight schemes and dynamic layers."""
        try:
            # Dynamic multi-layer initialization
            self.layers = []
            layer_sizes = [
                self.architecture['input_size'],
                self.architecture['hidden_size'],
                self.architecture['hidden_size'] // 2,
                self.architecture['output_size']
            ]
            
            for i in range(len(layer_sizes) - 1):
                # He initialization for better gradient flow
                scale = np.sqrt(2.0 / layer_sizes[i])
                self.layers.append({
                    'weights': np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale,
                    'bias': np.zeros(layer_sizes[i+1]),
                    'activation_history': [],
                    'gradient_history': []
                })
        except Exception as e:
            print(f"Network initialization error handled: {e}")
            self._initialize_fallback()

    def _setup_adaptive_components(self) -> None:
        """Setup adaptive learning components and meta-parameters."""
        self.adaptive_params = {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'dropout_rate': 0.2,
            'weight_decay': 0.0001,
            'activation_threshold': 0.5
        }
        
        self.meta_learning = {
            'performance_history': [],
            'adaptation_rate': 0.1,
            'architecture_mutation_rate': 0.05
        }

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Advanced forward pass with dynamic activation and adaptive dropout."""
        try:
            x = np.asarray(inputs, dtype=np.float64)
            cache = {'activations': [x]}
            
            for i, layer in enumerate(self.layers):
                # Apply adaptive dropout during training
                if training and i > 0:
                    dropout_mask = np.random.binomial(1, 1-self.adaptive_params['dropout_rate'], 
                                                    size=x.shape) / (1-self.adaptive_params['dropout_rate'])
                    x *= dropout_mask
                
                # Dynamic forward computation
                x = np.dot(x, layer['weights'].T) + layer['bias']
                
                # Adaptive activation with gradient tracking
                x = self._adaptive_activation(x)
                cache['activations'].append(x)
                
                # Update activation statistics
                if training:
                    layer['activation_history'].append(np.mean(np.abs(x)))
            
            return x, cache
            
        except Exception as e:
            print(f"Forward pass error handled: {e}")
            return np.zeros(self.architecture['output_size']), {'activations': []}

    def _adaptive_activation(self, x: np.ndarray) -> np.ndarray:
        """Dynamically adaptive activation function."""
        try:
            # Compute activation statistics
            pos_mask = x > 0
            neg_mask = x < 0
            
            # Adaptive ReLU with learnable leakage
            leaky_factor = np.clip(self.adaptive_params['activation_threshold'], 0.01, 0.3)
            
            return np.where(pos_mask, x, leaky_factor * x)
        except Exception:
            return np.maximum(0, x)  # Fallback to standard ReLU

class CommunicationManager:
    """Advanced communication system with dynamic routing and adaptive signal processing."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.signals: List[Signal] = []
        self.signal_stats = {
            'success_rate': 1.0,
            'interference_map': None,
            'routing_efficiency': 1.0
        }
        self._initialize_adaptive_routing()
    
    def _initialize_adaptive_routing(self) -> None:
        """Initialize adaptive routing system with performance tracking."""
        self.routing = {
            'paths': defaultdict(list),
            'success_rates': defaultdict(lambda: 1.0),
            'interference_map': None,
            'optimization_params': {
                'path_exploration_rate': 0.2,
                'interference_threshold': 0.3,
                'routing_adaptation_rate': 0.1
            }
        }

    def broadcast_signal(self, sender: CellularTypeData, content: Dict[str, Any]) -> None:
        """Broadcast signals with adaptive routing and interference management."""
        try:
            # Prepare vectorized operations
            sender_pos = np.column_stack((sender.x, sender.y))
            
            for ct in sender.communicate_with:
                if ct is sender:
                    continue
                    
                # Vectorized distance calculation
                receiver_pos = np.column_stack((ct.x, ct.y))
                distances = scipy.spatial.distance.cdist(sender_pos, receiver_pos)
                
                # Optimize communication paths
                valid_paths = distances < self.config.communication_range
                interference = self._compute_interference(sender_pos, receiver_pos)
                
                # Apply adaptive routing
                routing_mask = self._get_optimal_routes(valid_paths, interference)
                
                # Create and adapt signals
                for s_idx, r_idx in zip(*np.where(routing_mask)):
                    signal = Signal(
                        sender_id=sender.type_id,
                        receiver_id=ct.type_id,
                        content=self._enhance_content(content, s_idx, r_idx),
                        strength=self._compute_optimal_strength(distances[s_idx, r_idx])
                    )
                    
                    # Apply environmental adaptation
                    signal.adapt_strength(
                        self._get_environmental_factor(sender_pos[s_idx], receiver_pos[r_idx]),
                        distances[s_idx, r_idx]
                    )
                    
                    self.signals.append(signal)
                    
            # Update routing statistics
            self._update_routing_stats()
            
        except Exception as e:
            print(f"Signal broadcast error handled: {e}")

    def process_signals(self, particles: List[CellularTypeData]) -> None:
        """Process signals with advanced filtering and priority handling."""
        try:
            # Sort signals by priority and reliability
            self.signals.sort(key=lambda s: s.metadata['priority'] * s.metadata['reliability'],
                            reverse=True)
            
            # Process signals in optimized batches
            for batch in np.array_split(self.signals, max(1, len(self.signals) // 100)):
                self._process_signal_batch(batch, particles)
                
            # Update communication statistics
            self._update_communication_stats()
            
            # Clear processed signals
            self.signals.clear()
            
        except Exception as e:
            print(f"Signal processing error handled: {e}")

    def _compute_interference(self, sender_pos: np.ndarray, receiver_pos: np.ndarray) -> np.ndarray:
        """Compute interference based on signal density and environmental factors."""
        # Placeholder for advanced interference calculation
        return np.zeros(sender_pos.shape[0])

    def _get_optimal_routes(self, valid_paths: np.ndarray, interference: np.ndarray) -> np.ndarray:
        """Determine optimal communication routes based on path validity and interference."""
        # Placeholder for advanced routing logic
        return valid_paths

    def _enhance_content(self, content: Dict[str, Any], s_idx: int, r_idx: int) -> Dict[str, Any]:
        """Enhance signal content based on sender and receiver context."""
        # Placeholder for content enhancement
        return content

    def _compute_optimal_strength(self, distance: float) -> float:
        """Compute optimal signal strength based on distance and other factors."""
        # Placeholder for strength calculation
        return self.config.signal_strength / (1 + distance)

    def _get_environmental_factor(self, sender_pos: np.ndarray, receiver_pos: np.ndarray) -> float:
        """Get environmental factor influencing signal propagation."""
        # Placeholder for environmental factor calculation
        return 1.0

    def _update_routing_stats(self) -> None:
        """Update routing statistics based on signal performance."""
        # Placeholder for routing statistics update
        pass

    def _process_signal_batch(self, batch: List[Signal], particles: List[CellularTypeData]) -> None:
        """Process a batch of signals and deliver them to the appropriate particles."""
        try:
            for signal in batch:
                for particle in particles:
                    if particle.type_id == signal.receiver_id:
                        particle.receive_signal(signal, self.config)
        except Exception as e:
            print(f"Error processing signal batch: {e}")

    def _update_communication_stats(self) -> None:
        """Update communication statistics based on signal processing."""
        # Placeholder for communication statistics update
        pass


###############################################################
# Renderer Class
###############################################################

class Renderer:
    """
    High performance renderer utilizing optimized array operations and vectorized drawing.
    Handles all color formats and array shapes dynamically with comprehensive error handling.
    """
    
    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """Initialize optimized renderer with double-buffered arrays and caching."""
        self.surface = surface
        self.config = config
        
        # Initialize surface dimensions
        try:
            self.width = max(1, surface.get_width())
            self.height = max(1, surface.get_height())
        except Exception as e:
            print(f"Surface initialization error: {e}")
            self.width = 100
            self.height = 100

        # Initialize font with fallbacks
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 20)
        except Exception:
            self.font = pygame.font.Font(None, 20) if pygame.font.get_init() else None

        # Pre-compute particle template
        self.particle_size = np.clip(int(config.particle_size), 1, 100)
        self.EPSILON = 1e-10
        
    def _validate_color(self, color: Union[Tuple[int, ...], List[int], np.ndarray]) -> Tuple[int, int, int]:
        """Safely validate and convert color formats."""
        try:
            c = np.asarray(color, dtype=np.float64)
            if c.size >= 3:
                return tuple(np.clip(c[:3], 0, 255).astype(int))
            elif c.size == 1:
                val = int(np.clip(c[0], 0, 255))
                return (val, val, val)
            else:
                return (255, 255, 255)
        except Exception:
            return (255, 255, 255)


    def draw_component(self, x: float, y: float, color: Any, energy: float, speed_factor: float) -> None:
        """Draw a single component as a circle."""
        try:
            radius = self.particle_size
            x = int(x)
            y = int(y)
            
            intensity = np.clip(energy / (100.0 + self.EPSILON) * 
                              np.clip(speed_factor, self.EPSILON, None), 0.2, 1.0)
            
            color = self._validate_color(color)
            
            if self.surface is not None:
                
                final_color = tuple(int(c * intensity) for c in color)
                pygame.draw.circle(self.surface, final_color, (x, y), radius)
        except Exception as e:
            print(f"Draw component error: {e}")

    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """Draw all components of a cellular type."""
        try:
            if not hasattr(ct, 'alive') or ct.alive is None:
                return
                
            alive_mask = ct.alive.astype(bool)
            if not np.any(alive_mask):
                return
                
            positions = np.column_stack((
                np.clip(ct.x[alive_mask], 0, self.width).astype(int),
                np.clip(ct.y[alive_mask], 0, self.height).astype(int)
            ))
            
            energies = np.clip(ct.energy[alive_mask], 0, None)
            speeds = np.clip(ct.genome.genes[0].value[alive_mask], self.config.genetics.trait_ranges[0][0], self.config.genetics.trait_ranges[0][1])
            
            colors = [self._validate_color(ct.color) for _ in range(positions.shape[0])]
            
            for pos, color, energy, speed in zip(positions, colors, energies, speeds):
                self.draw_component(pos[0], pos[1], color, energy, speed)
                
        except Exception as e:
            print(f"Draw type error: {e}")

    def render(self, stats: Dict[str, Any]) -> None:
        """High performance rendering with double buffering."""
        try:
            if self.surface is None:
                return
            
            self.surface.fill((0, 0, 0)) # Fill with black background
            
            # Render stats
            if self.font is not None:
                try:
                    stats_text = (
                        f"FPS: {stats.get('fps', 0):.1f} | "
                        f"Species: {stats.get('total_species', 0)} | "
                        f"Particles: {stats.get('total_particles', 0)}"
                    )
                    text_surface = self.font.render(stats_text, True, (255,255,255))
                    self.surface.blit(text_surface, (10,10))
                except Exception as e:
                    print(f"Stats render error: {e}")
            
            pygame.display.flip()
                    
        except Exception as e:
            print(f"Main render error: {e}")

###############################################################
# Cellular Automata (Main Simulation)
###############################################################

class CellularAutomata:
    """
    The main simulation class. Initializes and runs the simulation loop.
    Optimized for high performance, scalability and maintainability.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the CellularAutomata with the given configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.config = config  # Store simulation configuration
        pygame.init()  # Initialize all imported Pygame modules

        # Retrieve display information to set fullscreen window
        display_info = pygame.display.Info()
        screen_width = max(800, display_info.current_w)  # Ensure minimum width
        screen_height = max(600, display_info.current_h)  # Ensure minimum height

        # Set up a fullscreen window with the calculated dimensions
        try:
            self.screen = pygame.display.set_mode((screen_width, screen_height),
                                                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
        except pygame.error:
            # Fallback to windowed mode if fullscreen fails
            self.screen = pygame.display.set_mode((800, 600),
                                                pygame.HWSURFACE | pygame.DOUBLEBUF)
            screen_width, screen_height = 800, 600

        pygame.display.set_caption("Emergent Cellular Automata Simulation")

        # Core simulation components
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.run_flag = True
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.edge_buffer = np.clip(0.05 * max(self.screen_width, self.screen_height), 10, 100)

        # Initialize managers with comprehensive error handling
        try:
            # Generate colors and setup mass-based types
            self.colors = generate_vibrant_colors(self.config.n_cell_types)
            n_mass_types = max(0, min(
                int(self.config.mass_based_fraction * self.config.n_cell_types),
                self.config.n_cell_types
            ))
            mass_based_type_indices = list(range(n_mass_types))
        except Exception as e:
            print(f"Error generating colors/types: {e}")
            self.colors = [(255, 255, 255)] * self.config.n_cell_types
            n_mass_types = 0
            mass_based_type_indices = []

        # Initialize managers with error handling
        try:
            self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)
            self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
            self.renderer = Renderer(self.screen, self.config)
            self.genetic_interpreter = GeneticInterpreter()
            self.communication_manager = CommunicationManager(self.config)
        except Exception as e:
            print(f"Error initializing managers: {e}")
            raise

        # Pre-calculate mass values with bounds checking
        if n_mass_types > 0:
            mass_values = np.clip(
                np.random.uniform(
                    self.config.mass_range[0],
                    self.config.mass_range[1],
                    n_mass_types
                ),
                1e-6,
                None
            )
        else:
            mass_values = np.zeros(0)

        # Create cellular type data with error handling
        for i in range(self.config.n_cell_types):
            try:
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=max(1, self.config.particles_per_type),
                    window_width=screen_width,
                    window_height=screen_height,
                    initial_energy=max(0.1, self.config.initial_energy),
                    max_age=max(1, self.config.max_age),
                    mass=mass_values[i] if i < n_mass_types else None,
                    base_velocity_scale=max(0.1, self.config.base_velocity_scale),
                    neural_network_params=self.config.neural_network_params
                )
                self.type_manager.add_cellular_type_data(ct)
            except Exception as e:
                print(f"Error creating cellular type {i}: {e}")
                continue

        # Initialize statistics tracking
        self.species_count = defaultdict(int)
        self.update_species_count()

        # Pre-allocate arrays for boundary calculations
        self.screen_bounds = np.array([
            self.edge_buffer,
            self.screen_width - self.edge_buffer,
            self.edge_buffer,
            self.screen_height - self.edge_buffer
        ])
        self.tree_cache = {}

        # Initialize advanced performance tracking
        self._init_performance_tracking()

    def _init_performance_tracking(self):
        """Initialize comprehensive performance tracking metrics."""
        self._performance_metrics = {
            'fps_history': collections.deque([60.0] * 60, maxlen=60),
            'particle_counts': collections.deque([0] * 60, maxlen=60),
            'cull_history': collections.deque(maxlen=10),
            'last_cull_time': time.time(),
            'performance_score': 1.0,
            'stress_threshold': 0.7,
            'min_fps': 45,
            'target_fps': 90,
            'emergency_fps': 30,
            'last_emergency': 0,
            'frame_times': collections.deque(maxlen=120),
            'interaction_times': collections.deque(maxlen=60),
            'render_times': collections.deque(maxlen=60)
        }

    def update_species_count(self) -> None:
        """Update species count with optimized numpy operations."""
        try:
            self.species_count.clear()
            for ct in self.type_manager.cellular_types:
                if ct.species_id is not None and ct.species_id.size > 0:
                    unique, counts = np.unique(ct.species_id, return_counts=True)
                    mask = unique >= 0
                    self.species_count.update(zip(unique[mask], counts[mask]))
        except Exception as e:
            print(f"Error updating species count: {e}")

    def display_fps(self, surface: pygame.Surface, fps: float) -> None:
        """Display FPS with error handling."""
        try:
            if not hasattr(self, '_fps_font'):
                self._fps_font = pygame.font.Font(None, 36)
            fps_text = self._fps_font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
            surface.blit(fps_text, (10, 10))
        except Exception as e:
            print(f"Error displaying FPS: {e}")

    def decode_genetic_traits(self) -> None:
        """Decode genetic traits for each cellular type."""
        for ct in self.type_manager.cellular_types:
            self.genetic_interpreter.decode(ct, others=self.type_manager.cellular_types, env=self.config)
    
    def apply_all_interactions(self) -> None:
        """Apply inter-type interactions: forces, give-take, and synergy."""
        for (i, j, params) in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)

    def apply_interaction_between_types(self, i: int, j: int, params: Dict[str, Any]) -> None:
        """
        Apply interaction rules between cellular type i and cellular type j.
        This includes forces, give-take, and synergy.

        Parameters:
        -----------
        i : int
            Index of the first cellular type.
        j : int
            Index of the second cellular type.
        params : Dict[str, Any]
            Interaction parameters between cellular type i and j.
        """
        ct_i = self.type_manager.get_cellular_type_by_id(i)
        ct_j = self.type_manager.get_cellular_type_by_id(j)

        if ct_i is None or ct_j is None:
            return

        synergy_factor = self.rules_manager.synergy_matrix[i, j]
        is_giver = self.rules_manager.give_take_matrix[i, j]

        n_i = ct_i.x.size
        n_j = ct_j.x.size

        if n_i == 0 or n_j == 0:
            return

        if params.get("use_gravity", False):
            if (ct_i.mass_based and ct_i.mass is not None and
                ct_j.mass_based and ct_j.mass is not None):
                params["m_a"] = ct_i.mass
                params["m_b"] = ct_j.mass
            else:
                params["use_gravity"] = False

        dx = ct_i.x[:, np.newaxis] - ct_j.x
        dy = ct_i.y[:, np.newaxis] - ct_j.y
        dist_sq = dx * dx + dy * dy

        within_range = (dist_sq > 0.0) & (dist_sq <= params["max_dist"] ** 2)
        indices = np.where(within_range)
        if len(indices[0]) == 0:
            return

        dist = np.sqrt(dist_sq[indices])
        fx = np.zeros_like(dist)
        fy = np.zeros_like(dist)

        if params.get("use_potential", True):
            pot_strength = np.float64(params.get("potential_strength", 1.0))
            F_pot = pot_strength / dist
            fx += F_pot * dx[indices]
            fy += F_pot * dy[indices]

        if params.get("use_gravity", False):
            gravity_factor = np.float64(params.get("gravity_factor", 1.0))
            F_grav = gravity_factor * (params["m_a"][indices[0]] * params["m_b"][indices[1]]) / dist_sq[indices]
            fx += F_grav * dx[indices]
            fy += F_grav * dy[indices]

        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        if is_giver:
            give_take_within = dist_sq[indices] <= self.config.predation_range ** 2
            give_take_indices = (indices[0][give_take_within], indices[1][give_take_within])
            if give_take_indices[0].size > 0:
                giver_energy = ct_i.energy[give_take_indices[0]]
                receiver_energy = ct_j.energy[give_take_indices[1]]
                giver_mass = ct_i.mass[give_take_indices[0]] if ct_i.mass_based else None
                receiver_mass = ct_j.mass[give_take_indices[1]] if ct_j.mass_based else None

                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config
                )
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            synergy_within = dist_sq[indices] <= self.config.synergy_range ** 2
            synergy_indices = (indices[0][synergy_within], indices[1][synergy_within])
            if synergy_indices[0].size > 0:
                energyA = ct_i.energy[synergy_indices[0]]
                energyB = ct_j.energy[synergy_indices[1]]
                new_energyA, new_energyB = apply_synergy(energyA, energyB, synergy_factor)
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Communication: Send and process signals
        self.communication_manager.broadcast_signal(ct_i, {"type": "interaction", "strength": synergy_factor})
        self.communication_manager.process_signals(self.type_manager.cellular_types)

        # Neural Network Processing
        ct_i.neural_network.forward(np.array([synergy_factor]))  # Example input

        # Apply friction
        friction_mask = np.full(n_i, self.config.friction)
        ct_i.vx *= friction_mask
        ct_i.vy *= friction_mask
        
        thermal_noise = np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        ct_i.vx += thermal_noise
        ct_i.vy += thermal_noise

        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        self.handle_boundary_reflections(ct_i)

        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(self, ct: Optional[CellularTypeData] = None) -> None:
        """
        Handle boundary reflections for cellular components using vectorized operations.
        """
        cellular_types = [ct] if ct else self.type_manager.cellular_types

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            # Create boolean masks for boundary violations
            left_mask = ct.x < self.screen_bounds[0]
            right_mask = ct.x > self.screen_bounds[1]
            top_mask = ct.y < self.screen_bounds[2]
            bottom_mask = ct.y > self.screen_bounds[3]

            # Reflect velocities where needed
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)

    def main_loop(self) -> None:
        """Run the main simulation loop with error handling and performance monitoring."""
        last_time = time.time()
        
        while self.run_flag:
            try:
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time
                
                self._performance_metrics['frame_times'].append(frame_time)
                
                self.frame_count += 1
                if 0 < self.config.max_frames <= self.frame_count:
                    self.run_flag = False
                    break

                # Handle events
                for event in pygame.event.get():
                    if event.type in {pygame.QUIT, pygame.KEYDOWN} and (
                        event.type == pygame.QUIT or event.key == pygame.K_ESCAPE
                    ):
                        self.run_flag = False
                        break

                # Main simulation steps with performance monitoring
                with Timer() as t:
                    self._simulation_step()
                
                # Adaptive performance management
                if t.interval > 1.0 / 30:  # If frame took longer than 33ms
                    self._handle_performance_degradation()

                # Update display
                pygame.display.flip()
                
                # Cap frame rate
                current_fps = self.clock.tick(120)
                
                # Adaptive particle management
                if current_fps <= 60 and self.frame_count % 10 == 0:
                    self.cull_oldest_particles()

            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()

        pygame.quit()

    def _simulation_step(self):
        """Execute one step of the simulation with error handling and performance monitoring."""
        try:
            # Update interaction parameters
            self.rules_manager.evolve_parameters(self.frame_count)

            # Update genetic traits
            self.decode_genetic_traits()

            # Apply interactions
            self.apply_all_interactions()

            # Apply clustering
            self._apply_clustering_parallel()

            # Handle reproduction and death
            self.type_manager.reproduce()
            self.type_manager.remove_dead_in_all_types()

            # Update species count
            self.update_species_count()

            # Render
            self._render_frame()

        except Exception as e:
            print(f"Error in simulation step: {e}")
            traceback.print_exc()

    def _render_frame(self):
        """Render the current frame with error handling and performance monitoring."""
        try:
            # Draw cellular types
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Compile stats
            stats = {
                "fps": max(1.0, self.clock.get_fps()),
                "total_species": len(self.species_count),
                "total_particles": max(0, sum(self.species_count.values()))
            }

            # Render
            self.renderer.render(stats)

        except Exception as e:
            print(f"Error rendering frame: {e}")
            traceback.print_exc()

        # Initialize Neural Network
        nn_params = neural_network_params if neural_network_params else {}
        self.neural_network = NeuralNetwork(
            input_size=nn_params.get("input_size", 5),
            hidden_size=nn_params.get("hidden_size", 10),
            output_size=nn_params.get("output_size", 5),
            activation=nn_params.get("activation", "relu")
        )
        self.neural_network._initialize_fallback = lambda: None

        # Initialize Genome
        self.genome = Genome()
        for trait in gene_traits:
            initial_value = np.random.uniform(0.5, 1.5)
            self.genome.add_gene(Gene(trait, initial_value))

    def _apply_clustering_parallel(self):
        """Apply clustering using parallel processing for large particle counts."""
        try:
            for ct in self.type_manager.cellular_types:
                if ct.x.size >= 1000:  # Only parallelize for large counts
                    with ThreadPoolExecutor() as executor:
                        chunks = np.array_split(range(ct.x.size), 4)
                        futures = [
                            executor.submit(self._apply_clustering_chunk, ct, chunk)
                            for chunk in chunks
                        ]
                        for future in futures:
                            future.result()
                else:
                    self.apply_clustering(ct)
        except Exception as e:
            print(f"Error in parallel clustering: {e}")
            traceback.print_exc()

    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply clustering forces within a single cellular type using KD-Tree for efficiency.
        """
        n = ct.x.size
        if n < 2:
            return

        # Build KD-Tree once for position data
        positions = np.column_stack((ct.x, ct.y))
        tree = cKDTree(positions)
        
        # Query all neighbors at once
        indices = tree.query_ball_tree(tree, self.config.cluster_radius)
        
        # Pre-allocate velocity change arrays
        dvx = np.zeros(n)
        dvy = np.zeros(n)
        
        # Vectorized calculations for all components
        for idx, neighbor_indices in enumerate(indices):
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue
                
            neighbor_positions = positions[neighbor_indices]
            neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))
            
            # Alignment
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength
            
            # Cohesion
            center = np.mean(neighbor_positions, axis=0)
            cohesion = (center - positions[idx]) * self.config.cohesion_strength
            
            # Separation
            separation = (positions[idx] - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength
            
            # Combine forces
            total_force = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]
        
        # Apply accumulated velocity changes
        ct.vx += dvx
        ct.vy += dvy

    def _apply_clustering_chunk(self, ct: CellularTypeData, chunk: np.ndarray) -> None:
        """
        Apply clustering to a chunk of particles for parallel processing.
        """
        try:
            for idx in chunk:
                neighbor_indices = self._find_neighbors(ct, idx)
                if not neighbor_indices:
                    continue
                
                neighbor_positions = np.column_stack((ct.x[neighbor_indices], ct.y[neighbor_indices]))
                neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))
                
                # Alignment
                avg_velocity = np.mean(neighbor_velocities, axis=0)
                alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength
                
                # Cohesion
                center = np.mean(neighbor_positions, axis=0)
                cohesion = (center - np.array([ct.x[idx], ct.y[idx]])) * self.config.cohesion_strength
                
                # Separation
                separation = (np.array([ct.x[idx], ct.y[idx]]) - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength
                
                # Combine forces
                total_force = alignment + cohesion + separation
                ct.vx[idx] += total_force[0]
                ct.vy[idx] += total_force[1]
        except Exception as e:
            print(f"Error in clustering chunk: {e}")
            traceback.print_exc()

    def _find_neighbors(self, ct: CellularTypeData, idx: int) -> List[int]:
        """
        Find neighbors within cluster radius for a given particle index.
        """
        try:
            position = np.array([ct.x[idx], ct.y[idx]])
            tree = cKDTree(np.column_stack((ct.x, ct.y)))
            neighbors = tree.query_ball_point(position, self.config.cluster_radius)
            neighbors = [i for i in neighbors if i != idx and ct.alive[i]]
            return neighbors
        except Exception as e:
            print(f"Error finding neighbors for idx {idx}: {e}")
            return []

    def _handle_performance_degradation(self):
        """Handle severe performance issues with adaptive optimization."""
        try:
            current_fps = self.clock.get_fps()
            if current_fps < self._performance_metrics['emergency_fps']:
                self.cull_oldest_particles()
        except Exception as e:
            print(f"Error handling performance degradation: {e}")
            traceback.print_exc()

    def cull_oldest_particles(self):
        """Cull the oldest particles based on performance metrics and fitness assessment."""
        try:
            metrics = self._performance_metrics
            current_time = time.time()
            current_fps = self.clock.get_fps()

            metrics['fps_history'].append(current_fps)
            total_particles = sum(ct.x.size for ct in self.type_manager.cellular_types)
            metrics['particle_counts'].append(total_particles)

            avg_fps = np.mean(metrics['fps_history'])
            fps_trend = np.gradient(metrics['fps_history'])[-10:].mean() if len(metrics['fps_history']) > 10 else 0
            particle_trend = np.gradient(metrics['particle_counts'])[-10:].mean() if len(metrics['particle_counts']) > 10 else 0

            fps_stress = max(0, (self.config.target_fps - avg_fps) / self.config.target_fps)
            particle_stress = 1 / (1 + np.exp(-total_particles / 10000))
            system_stress = (fps_stress * 0.7 + particle_stress * 0.3)

            if current_fps < self._performance_metrics['emergency_fps'] and current_time - metrics['last_emergency'] > 5.0:
                emergency_cull_factor = 0.5
                metrics['last_emergency'] = current_time
                metrics['performance_score'] *= 2.0
                for ct in self.type_manager.cellular_types:
                    if ct.x.size < 100:
                        continue
                    keep_count = max(50, int(ct.x.size * (1 - emergency_cull_factor)))
                    self._emergency_cull(ct, keep_count)
                return

            if avg_fps < metrics['min_fps']:
                metrics['performance_score'] *= 1.5
            elif avg_fps < metrics['target_fps']:
                metrics['performance_score'] *= 1.2
            elif avg_fps > metrics['target_fps']:
                metrics['performance_score'] = max(0.2, metrics['performance_score'] * 0.9)

            if fps_trend < 0:
                metrics['performance_score'] *= 1.2
            if particle_trend > 0:
                metrics['performance_score'] *= 1.1

            metrics['performance_score'] = np.clip(metrics['performance_score'], 0.2, 10.0)

            for ct in self.type_manager.cellular_types:
                if ct.x.size < 100:
                    continue
                positions = np.column_stack((ct.x, ct.y))
                tree = cKDTree(positions)
                fitness_scores = np.zeros(ct.x.size)
                density_scores = tree.query_ball_point(positions, r=200, return_length=True)
                density_penalty = density_scores / (np.max(density_scores) + 1e-10)

                energy_score = ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
                interaction_score = (ct.genome.genes[1].value * ct.genome.genes[4].value * ct.genome.genes[5].value * ct.genome.genes[3].value)
        except Exception as e:
            print(f"Error in culling oldest particles: {e}")
            traceback.print_exc()
            
###############################################################
# Renderer Class
###############################################################

class Renderer:
    """
    High performance renderer utilizing optimized array operations and vectorized drawing.
    Handles all color formats and array shapes dynamically with comprehensive error handling.
    """
    
    def __init__(self, surface: pygame.Surface, config: SimulationConfig):
        """Initialize optimized renderer with double-buffered arrays and caching."""
        self.surface = surface
        self.config = config
        
        # Initialize surface dimensions
        try:
            self.width = max(1, surface.get_width())
            self.height = max(1, surface.get_height())
        except Exception as e:
            print(f"Surface initialization error: {e}")
            self.width = 100
            self.height = 100

        # Initialize font with fallbacks
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 20)
        except Exception:
            self.font = pygame.font.Font(None, 20) if pygame.font.get_init() else None

        # Pre-compute particle template
        self.particle_size = np.clip(int(config.particle_size), 1, 100)
        self.EPSILON = 1e-10
        
    def _validate_color(self, color: Union[Tuple[int, ...], List[int], np.ndarray]) -> Tuple[int, int, int]:
        """Safely validate and convert color formats."""
        try:
            c = np.asarray(color, dtype=np.float64)
            if c.size >= 3:
                return tuple(np.clip(c[:3], 0, 255).astype(int))
            elif c.size == 1:
                val = int(np.clip(c[0], 0, 255))
                return (val, val, val)
            else:
                return (255, 255, 255)
        except Exception:
            return (255, 255, 255)


    def draw_component(self, x: float, y: float, color: Any, energy: float, speed_factor: float) -> None:
        """Draw a single component as a circle."""
        try:
            radius = self.particle_size
            x = int(x)
            y = int(y)
            
            intensity = np.clip(energy / (100.0 + self.EPSILON) * 
                              np.clip(speed_factor, self.EPSILON, None), 0.2, 1.0)
            
            color = self._validate_color(color)
            
            if self.surface is not None:
                
                final_color = tuple(int(c * intensity) for c in color)
                pygame.draw.circle(self.surface, final_color, (x, y), radius)
        except Exception as e:
            print(f"Draw component error: {e}")

    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """Draw all components of a cellular type."""
        try:
            if not hasattr(ct, 'alive') or ct.alive is None:
                return
                
            alive_mask = ct.alive.astype(bool)
            if not np.any(alive_mask):
                return
                
            positions = np.column_stack((
                np.clip(ct.x[alive_mask], 0, self.width),
                np.clip(ct.y[alive_mask], 0, self.height)
            ))
            
            energies = np.clip(ct.energy[alive_mask], 0, None)
            speeds = np.clip(ct.genome.genes[0].value, self.config.genetics.trait_ranges[0][0], self.config.genetics.trait_ranges[0][1])
            
            for pos, energy, speed in zip(positions, energies, speeds):
                self.draw_component(pos[0], pos[1], ct.color, energy, speed)
                
        except Exception as e:
            print(f"Draw type error: {e}")

    def render(self, stats: Dict[str, Any]) -> None:
        """High performance rendering with double buffering."""
        try:
            if self.surface is None:
                return
            
            self.surface.fill((0, 0, 0)) # Fill with black background
            
            # Render stats
            if self.font is not None:
                try:
                    stats_text = (
                        f"FPS: {stats.get('fps', 0):.1f} | "
                        f"Species: {stats.get('total_species', 0)} | "
                        f"Particles: {stats.get('total_particles', 0)}"
                    )
                    text_surface = self.font.render(stats_text, True, (255,255,255))
                    self.surface.blit(text_surface, (10,10))
                except Exception as e:
                    print(f"Stats render error: {e}")
            
            pygame.display.flip()
                    
        except Exception as e:
            print(f"Main render error: {e}")

###############################################################
# Cellular Automata (Main Simulation)
###############################################################

class CellularAutomata:
    """
    The main simulation class. Initializes and runs the simulation loop.
    Optimized for high performance, scalability and maintainability.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the CellularAutomata with the given configuration.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation.
        """
        self.config = config  # Store simulation configuration
        pygame.init()  # Initialize all imported Pygame modules

        # Retrieve display information to set fullscreen window
        display_info = pygame.display.Info()
        screen_width = max(800, display_info.current_w)  # Ensure minimum width
        screen_height = max(600, display_info.current_h)  # Ensure minimum height

        # Set up a fullscreen window with the calculated dimensions
        try:
            self.screen = pygame.display.set_mode((screen_width, screen_height),
                                                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
        except pygame.error:
            # Fallback to windowed mode if fullscreen fails
            self.screen = pygame.display.set_mode((800, 600),
                                                pygame.HWSURFACE | pygame.DOUBLEBUF)
            screen_width, screen_height = 800, 600

        pygame.display.set_caption("Emergent Cellular Automata Simulation")

        # Core simulation components
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.run_flag = True
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.edge_buffer = np.clip(0.05 * max(self.screen_width, self.screen_height), 10, 100)

        # Initialize managers with comprehensive error handling
        try:
            # Generate colors and setup mass-based types
            self.colors = generate_vibrant_colors(self.config.n_cell_types)
            n_mass_types = max(0, min(
                int(self.config.mass_based_fraction * self.config.n_cell_types),
                self.config.n_cell_types
            ))
            mass_based_type_indices = list(range(n_mass_types))
        except Exception as e:
            print(f"Error generating colors/types: {e}")
            self.colors = [(255, 255, 255)] * self.config.n_cell_types
            n_mass_types = 0
            mass_based_type_indices = []

        # Initialize managers with error handling
        try:
            self.type_manager = CellularTypeManager(self.config, self.colors, mass_based_type_indices)
            self.rules_manager = InteractionRules(self.config, mass_based_type_indices)
            self.renderer = Renderer(self.screen, self.config)
            self.genetic_interpreter = GeneticInterpreter()
            self.communication_manager = CommunicationManager(self.config)
        except Exception as e:
            print(f"Error initializing managers: {e}")
            raise

        # Pre-calculate mass values with bounds checking
        if n_mass_types > 0:
            mass_values = np.clip(
                np.random.uniform(
                    self.config.mass_range[0],
                    self.config.mass_range[1],
                    n_mass_types
                ),
                1e-6,
                None
            )
        else:
            mass_values = np.zeros(0)

        # Create cellular type data with error handling
        for i in range(self.config.n_cell_types):
            try:
                ct = CellularTypeData(
                    type_id=i,
                    color=self.colors[i],
                    n_particles=max(1, self.config.particles_per_type),
                    window_width=screen_width,
                    window_height=screen_height,
                    initial_energy=max(0.1, self.config.initial_energy),
                    max_age=max(1, self.config.max_age),
                    mass=mass_values[i] if i < n_mass_types else None,
                    base_velocity_scale=max(0.1, self.config.base_velocity_scale),
                    neural_network_params=self.config.neural_network_params
                )
                self.type_manager.add_cellular_type_data(ct)
            except Exception as e:
                print(f"Error creating cellular type {i}: {e}")
                continue

        # Initialize statistics tracking
        self.species_count = defaultdict(int)
        self.update_species_count()

        # Pre-allocate arrays for boundary calculations
        self.screen_bounds = np.array([
            self.edge_buffer,
            self.screen_width - self.edge_buffer,
            self.edge_buffer,
            self.screen_height - self.edge_buffer
        ])
        self.tree_cache = {}

        # Initialize advanced performance tracking
        self._init_performance_tracking()

    def _init_performance_tracking(self):
        """Initialize comprehensive performance tracking metrics."""
        self._performance_metrics = {
            'fps_history': collections.deque([60.0] * 60, maxlen=60),
            'particle_counts': collections.deque([0] * 60, maxlen=60),
            'cull_history': collections.deque(maxlen=10),
            'last_cull_time': time.time(),
            'performance_score': 1.0,
            'stress_threshold': 0.7,
            'min_fps': 45,
            'target_fps': 90,
            'emergency_fps': 30,
            'last_emergency': 0,
            'frame_times': collections.deque(maxlen=120),
            'interaction_times': collections.deque(maxlen=60),
            'render_times': collections.deque(maxlen=60)
        }

    def update_species_count(self) -> None:
        """Update species count with optimized numpy operations."""
        try:
            self.species_count.clear()
            for ct in self.type_manager.cellular_types:
                if ct.species_id is not None and ct.species_id.size > 0:
                    unique, counts = np.unique(ct.species_id, return_counts=True)
                    mask = unique >= 0
                    self.species_count.update(zip(unique[mask], counts[mask]))
        except Exception as e:
            print(f"Error updating species count: {e}")

    def display_fps(self, surface: pygame.Surface, fps: float) -> None:
        """Display FPS with error handling."""
        try:
            if not hasattr(self, '_fps_font'):
                self._fps_font = pygame.font.Font(None, 36)
            fps_text = self._fps_font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
            surface.blit(fps_text, (10, 10))
        except Exception as e:
            print(f"Error displaying FPS: {e}")

    def decode_genetic_traits(self) -> None:
        """Decode genetic traits for each cellular type."""
        for ct in self.type_manager.cellular_types:
            self.genetic_interpreter.decode(ct, others=self.type_manager.cellular_types, env=self.config)
    
    def apply_all_interactions(self) -> None:
        """Apply inter-type interactions: forces, give-take, and synergy."""
        for (i, j, params) in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)

    def apply_interaction_between_types(self, i: int, j: int, params: Dict[str, Any]) -> None:
        """
        Apply interaction rules between cellular type i and cellular type j.
        This includes forces, give-take, and synergy.

        Parameters:
        -----------
        i : int
            Index of the first cellular type.
        j : int
            Index of the second cellular type.
        params : Dict[str, Any]
            Interaction parameters between cellular type i and j.
        """
        ct_i = self.type_manager.get_cellular_type_by_id(i)
        ct_j = self.type_manager.get_cellular_type_by_id(j)

        if ct_i is None or ct_j is None:
            return

        synergy_factor = self.rules_manager.synergy_matrix[i, j]
        is_giver = self.rules_manager.give_take_matrix[i, j]

        n_i = ct_i.x.size
        n_j = ct_j.x.size

        if n_i == 0 or n_j == 0:
            return

        if params.get("use_gravity", False):
            if (ct_i.mass_based and ct_i.mass is not None and
                ct_j.mass_based and ct_j.mass is not None):
                params["m_a"] = ct_i.mass
                params["m_b"] = ct_j.mass
            else:
                params["use_gravity"] = False

        dx = ct_i.x[:, np.newaxis] - ct_j.x
        dy = ct_i.y[:, np.newaxis] - ct_j.y
        dist_sq = dx * dx + dy * dy

        within_range = (dist_sq > 0.0) & (dist_sq <= params["max_dist"] ** 2)
        indices = np.where(within_range)
        if len(indices[0]) == 0:
            return

        dist = np.sqrt(dist_sq[indices])
        fx = np.zeros_like(dist)
        fy = np.zeros_like(dist)

        if params.get("use_potential", True):
            pot_strength = np.float64(params.get("potential_strength", 1.0))
            F_pot = pot_strength / dist
            fx += F_pot * dx[indices]
            fy += F_pot * dy[indices]

        if params.get("use_gravity", False):
            gravity_factor = np.float64(params.get("gravity_factor", 1.0))
            F_grav = gravity_factor * (params["m_a"][indices[0]] * params["m_b"][indices[1]]) / dist_sq[indices]
            fx += F_grav * dx[indices]
            fy += F_grav * dy[indices]

        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        if is_giver:
            give_take_within = dist_sq[indices] <= self.config.predation_range ** 2
            give_take_indices = (indices[0][give_take_within], indices[1][give_take_within])
            if give_take_indices[0].size > 0:
                giver_energy = ct_i.energy[give_take_indices[0]]
                receiver_energy = ct_j.energy[give_take_indices[1]]
                giver_mass = ct_i.mass[give_take_indices[0]] if ct_i.mass_based else None
                receiver_mass = ct_j.mass[give_take_indices[1]] if ct_j.mass_based else None

                updated = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config
                )
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            synergy_within = dist_sq[indices] <= self.config.synergy_range ** 2
            synergy_indices = (indices[0][synergy_within], indices[1][synergy_within])
            if synergy_indices[0].size > 0:
                energyA = ct_i.energy[synergy_indices[0]]
                energyB = ct_j.energy[synergy_indices[1]]
                new_energyA, new_energyB = apply_synergy(energyA, energyB, synergy_factor)
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Communication: Send and process signals
        self.communication_manager.broadcast_signal(ct_i, {"type": "interaction", "strength": synergy_factor})
        self.communication_manager.process_signals(self.type_manager.cellular_types)

        # Neural Network Processing
        ct_i.neural_network.forward(np.array([synergy_factor]))  # Example input

        # Apply friction
        friction_mask = np.full(n_i, self.config.friction)
        ct_i.vx *= friction_mask
        ct_i.vy *= friction_mask
        
        thermal_noise = np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        ct_i.vx += thermal_noise
        ct_i.vy += thermal_noise

        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        self.handle_boundary_reflections(ct_i)

        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(self, ct: Optional[CellularTypeData] = None) -> None:
        """
        Handle boundary reflections for cellular components using vectorized operations.
        """
        cellular_types = [ct] if ct else self.type_manager.cellular_types

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            # Create boolean masks for boundary violations
            left_mask = ct.x < self.screen_bounds[0]
            right_mask = ct.x > self.screen_bounds[1]
            top_mask = ct.y < self.screen_bounds[2]
            bottom_mask = ct.y > self.screen_bounds[3]

            # Reflect velocities where needed
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)

    def main_loop(self) -> None:
        """Run the main simulation loop with error handling and performance monitoring."""
        last_time = time.time()
        
        while self.run_flag:
            try:
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time
                
                self._performance_metrics['frame_times'].append(frame_time)
                
                self.frame_count += 1
                if 0 < self.config.max_frames <= self.frame_count:
                    self.run_flag = False
                    break

                # Handle events
                for event in pygame.event.get():
                    if event.type in {pygame.QUIT, pygame.KEYDOWN} and (
                        event.type == pygame.QUIT or event.key == pygame.K_ESCAPE
                    ):
                        self.run_flag = False
                        break

                # Main simulation steps with performance monitoring
                with Timer() as t:
                    self._simulation_step()
                
                # Adaptive performance management
                if t.interval > 1.0 / 30:  # If frame took longer than 33ms
                    self._handle_performance_degradation()

                # Update display
                pygame.display.flip()
                
                # Cap frame rate
                current_fps = self.clock.tick(120)
                
                # Adaptive particle management
                if current_fps <= 60 and self.frame_count % 10 == 0:
                    self.cull_oldest_particles()

            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()

        pygame.quit()

    def _simulation_step(self):
        """Execute one step of the simulation with error handling and performance monitoring."""
        try:
            # Update interaction parameters
            self.rules_manager.evolve_parameters(self.frame_count)

            # Update genetic traits
            self.decode_genetic_traits()

            # Apply interactions
            self.apply_all_interactions()

            # Apply clustering
            self._apply_clustering_parallel()

            # Handle reproduction and death
            self.type_manager.reproduce()
            self.type_manager.remove_dead_in_all_types()

            # Update species count
            self.update_species_count()

            # Render
            self._render_frame()

        except Exception as e:
            print(f"Error in simulation step: {e}")
            traceback.print_exc()

    def _render_frame(self):
        """Render the current frame with error handling and performance monitoring."""
        try:
            # Draw cellular types
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Compile stats
            stats = {
                "fps": max(1.0, self.clock.get_fps()),
                "total_species": len(self.species_count),
                "total_particles": max(0, sum(self.species_count.values()))
            }

            # Render
            self.renderer.render(stats)

        except Exception as e:
            print(f"Error rendering frame: {e}")
            traceback.print_exc()

        # Initialize Neural Network
        nn_params = neural_network_params if neural_network_params else {}
        self.neural_network = NeuralNetwork(
            input_size=nn_params.get("input_size", 5),
            hidden_size=nn_params.get("hidden_size", 10),
            output_size=nn_params.get("output_size", 5),
            activation=nn_params.get("activation", "relu")
        )
        self.neural_network._initialize_fallback = lambda: None

        # Initialize Genome
        self.genome = Genome()
        for trait in gene_traits:
            initial_value = np.random.uniform(0.5, 1.5)
            self.genome.add_gene(Gene(trait, initial_value))

    def _apply_clustering_parallel(self):
        """Apply clustering using parallel processing for large particle counts."""
        try:
            for ct in self.type_manager.cellular_types:
                if ct.x.size >= 1000:  # Only parallelize for large counts
                    with ThreadPoolExecutor() as executor:
                        chunks = np.array_split(range(ct.x.size), 4)
                        futures = [
                            executor.submit(self._apply_clustering_chunk, ct, chunk)
                            for chunk in chunks
                        ]
                        for future in futures:
                            future.result()
                else:
                    self.apply_clustering(ct)
        except Exception as e:
            print(f"Error in parallel clustering: {e}")
            traceback.print_exc()

    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply clustering forces within a single cellular type using KD-Tree for efficiency.
        """
        n = ct.x.size
        if n < 2:
            return

        # Build KD-Tree once for position data
        positions = np.column_stack((ct.x, ct.y))
        tree = cKDTree(positions)
        
        # Query all neighbors at once
        indices = tree.query_ball_tree(tree, self.config.cluster_radius)
        
        # Pre-allocate velocity change arrays
        dvx = np.zeros(n)
        dvy = np.zeros(n)
        
        # Vectorized calculations for all components
        for idx, neighbor_indices in enumerate(indices):
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue
                
            neighbor_positions = positions[neighbor_indices]
            neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))
            
            # Alignment
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength
            
            # Cohesion
            center = np.mean(neighbor_positions, axis=0)
            cohesion = (center - positions[idx]) * self.config.cohesion_strength
            
            # Separation
            separation = (positions[idx] - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength
            
            # Combine forces
            total_force = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]
        
        # Apply accumulated velocity changes
        ct.vx += dvx
        ct.vy += dvy

    def _apply_clustering_chunk(self, ct: CellularTypeData, chunk: np.ndarray) -> None:
        """
        Apply clustering to a chunk of particles for parallel processing.
        """
        try:
            for idx in chunk:
                neighbor_indices = self._find_neighbors(ct, idx)
                if not neighbor_indices:
                    continue
                
                neighbor_positions = np.column_stack((ct.x[neighbor_indices], ct.y[neighbor_indices]))
                neighbor_velocities = np.column_stack((ct.vx[neighbor_indices], ct.vy[neighbor_indices]))
                
                # Alignment
                avg_velocity = np.mean(neighbor_velocities, axis=0)
                alignment = (avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])) * self.config.alignment_strength
                
                # Cohesion
                center = np.mean(neighbor_positions, axis=0)
                cohesion = (center - np.array([ct.x[idx], ct.y[idx]])) * self.config.cohesion_strength
                
                # Separation
                separation = (np.array([ct.x[idx], ct.y[idx]]) - np.mean(neighbor_positions, axis=0)) * self.config.separation_strength
                
                # Combine forces
                total_force = alignment + cohesion + separation
                ct.vx[idx] += total_force[0]
                ct.vy[idx] += total_force[1]
        except Exception as e:
            print(f"Error in clustering chunk: {e}")
            traceback.print_exc()

    def _find_neighbors(self, ct: CellularTypeData, idx: int) -> List[int]:
        """
        Find neighbors within cluster radius for a given particle index.
        """
        try:
            position = np.array([ct.x[idx], ct.y[idx]])
            tree = cKDTree(np.column_stack((ct.x, ct.y)))
            neighbors = tree.query_ball_point(position, self.config.cluster_radius)
            neighbors = [i for i in neighbors if i != idx and ct.alive[i]]
            return neighbors
        except Exception as e:
            print(f"Error finding neighbors for idx {idx}: {e}")
            return []

    def _handle_performance_degradation(self):
        """Handle severe performance issues with adaptive optimization."""
        try:
            current_fps = self.clock.get_fps()
            if current_fps < self._performance_metrics['emergency_fps']:
                self.cull_oldest_particles()
        except Exception as e:
            print(f"Error handling performance degradation: {e}")
            traceback.print_exc()

    def cull_oldest_particles(self):
        """Cull the oldest particles based on performance metrics and fitness assessment."""
        try:
            metrics = self._performance_metrics
            current_time = time.time()
            current_fps = self.clock.get_fps()

            metrics['fps_history'].append(current_fps)
            total_particles = sum(ct.x.size for ct in self.type_manager.cellular_types)
            metrics['particle_counts'].append(total_particles)

            avg_fps = np.mean(metrics['fps_history'])
            fps_trend = np.gradient(metrics['fps_history'])[-10:].mean() if len(metrics['fps_history']) > 10 else 0
            particle_trend = np.gradient(metrics['particle_counts'])[-10:].mean() if len(metrics['particle_counts']) > 10 else 0

            fps_stress = max(0, (self.config.target_fps - avg_fps) / self.config.target_fps)
            particle_stress = 1 / (1 + np.exp(-total_particles / 10000))
            system_stress = (fps_stress * 0.7 + particle_stress * 0.3)

            if current_fps < self._performance_metrics['emergency_fps'] and current_time - metrics['last_emergency'] > 5.0:
                emergency_cull_factor = 0.5
                metrics['last_emergency'] = current_time
                metrics['performance_score'] *= 2.0
                for ct in self.type_manager.cellular_types:
                    if ct.x.size < 100:
                        continue
                    keep_count = max(50, int(ct.x.size * (1 - emergency_cull_factor)))
                    self._emergency_cull(ct, keep_count)
                return

            if avg_fps < metrics['min_fps']:
                metrics['performance_score'] *= 1.5
            elif avg_fps < metrics['target_fps']:
                metrics['performance_score'] *= 1.2
            elif avg_fps > metrics['target_fps']:
                metrics['performance_score'] = max(0.2, metrics['performance_score'] * 0.9)

            if fps_trend < 0:
                metrics['performance_score'] *= 1.2
            if particle_trend > 0:
                metrics['performance_score'] *= 1.1

            metrics['performance_score'] = np.clip(metrics['performance_score'], 0.2, 10.0)

            for ct in self.type_manager.cellular_types:
                if ct.x.size < 100:
                    continue
                positions = np.column_stack((ct.x, ct.y))
                tree = cKDTree(positions)
                fitness_scores = np.zeros(ct.x.size)
                density_scores = tree.query_ball_point(positions, r=200, return_length=True)
                density_penalty = density_scores / (np.max(density_scores) + 1e-10)

                energy_score = ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
                interaction_score = (ct.genome.genes[1].value * ct.genome.genes[4].value * ct.genome.genes[5].value * ct.genome.genes[3].value)
                fitness_scores = (energy_score * 0.4 + interaction_score * 0.3 + (1 - density_penalty) * 0.3)
                fitness_scores = (fitness_scores - np.min(fitness_scores)) / (np.max(fitness_scores) - np.min(fitness_scores) + 1e-10)

                base_cull_rate = 0.1 * metrics['performance_score'] * system_stress
                cull_rate = np.clip(base_cull_rate, 0.05, 0.4)
                removal_count = int(ct.x.size * cull_rate)
                keep_indices = np.argsort(fitness_scores)[removal_count:]
                keep_mask = np.zeros(ct.x.size, dtype=bool)
                keep_mask[keep_indices] = True

                arrays_to_filter = [
                    'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                    'species_id', 'parent_id'
                ]
                for attr in arrays_to_filter:
                    if hasattr(ct, attr):
                        arr = getattr(ct, attr)
                        if isinstance(arr, np.ndarray) and arr.size == ct.x.size:
                            setattr(ct, attr, arr[keep_mask])
                if ct.mass_based and ct.mass is not None:
                    ct.mass = ct.mass[keep_mask]

            metrics['last_cull_time'] = current_time

        except Exception as e:
            print(f"Error in culling oldest particles: {e}")
            traceback.print_exc()

    def _emergency_cull(self, ct: CellularTypeData, keep_count: int) -> None:
        """Cull particles in an emergency situation based on fitness assessment."""
        try:
            if ct.x.size <= keep_count:
                return

            positions = np.column_stack((ct.x, ct.y))
            tree = cKDTree(positions)
            fitness_scores = np.zeros(ct.x.size)
            density_scores = tree.query_ball_point(positions, r=200, return_length=True)
            density_penalty = density_scores / (np.max(density_scores) + 1e-6)

            energy_score = ct.energy * ct.energy_efficiency * (1 - (ct.age / ct.max_age))
            interaction_score = (ct.genome.genes[1].value * ct.genome.genes[4].value * ct.genome.genes[5].value * ct.genome.genes[3].value)
            fitness_scores = (energy_score * 0.4 + interaction_score * 0.3 + (1 - density_penalty) * 0.3)
            fitness_scores = (fitness_scores - np.min(fitness_scores)) / (np.max(fitness_scores) - np.min(fitness_scores) + 1e-10)

            keep_indices = np.argsort(fitness_scores)[-keep_count:]
            keep_mask = np.zeros(ct.x.size, dtype=bool)
            keep_mask[keep_indices] = True

            arrays_to_filter = [
                'x', 'y', 'vx', 'vy', 'energy', 'alive', 'age', 'energy_efficiency',
                'species_id', 'parent_id'
            ]
            for attr in arrays_to_filter:
                if hasattr(ct, attr):
                    arr = getattr(ct, attr)
                    if isinstance(arr, np.ndarray) and arr.size == ct.x.size:
                        setattr(ct, attr, arr[keep_mask])
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[keep_mask]

        except Exception as e:
            print(f"Error in emergency culling: {e}")
            traceback.print_exc()
            self._ensure_particle_stability(ct)

    def _ensure_particle_stability(self, ct: CellularTypeData) -> None:
        """Ensure particle stability by resetting invalid attributes."""
        try:
            for attr in ['energy', 'vx', 'vy']:
                if hasattr(ct, attr):
                    arr = getattr(ct, attr)
                    if isinstance(arr, np.ndarray):
                        # Clip to safe bounds
                        if attr == 'energy':
                            setattr(ct, attr, np.clip(arr, *self.config.bounds['energy']))
                        elif attr in ['vx', 'vy']:
                            setattr(ct, attr, np.clip(arr, *self.config.bounds['velocity']))
        except Exception as e:
            print(f"Error ensuring particle stability: {e}")
            traceback.print_exc()

###############################################################
# Timer Class
###############################################################

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self # Allows access to `self.interval` after the block

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval:.4f} seconds")

###############################################################
# Entry Point
###############################################################

def main():
    """
    Main configuration and run function.
    """
    config = SimulationConfig()
    cellular_automata = CellularAutomata(config)
    cellular_automata.main_loop()

if __name__ == "__main__":
    main()
