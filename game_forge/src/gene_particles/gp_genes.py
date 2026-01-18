from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

import numpy as np

# Use TYPE_CHECKING for circular imports
if TYPE_CHECKING:
    from game_forge.src.gene_particles.gp_config import (
        GeneticParamConfig,
        SimulationConfig,
    )
    from game_forge.src.gene_particles.gp_types import CellularTypeData

from game_forge.src.gene_particles.gp_types import BoolArray, FloatArray, GeneData
from game_forge.src.gene_particles.gp_utility import mutate_trait


class PredationStrategy(Enum):
    """Predation strategies used by cellular entities."""

    OPPORTUNISTIC = auto()  # Attack when opportunity arises
    ENERGY_OPTIMAL = auto()  # Attack only when energy benefit exceeds cost
    SIZE_BASED = auto()  # Attack smaller entities preferentially
    TERRITORIAL = auto()  # Attack entities in proximity to territory


def apply_movement_gene(
    particle: "CellularTypeData", gene_data: GeneData, env: "SimulationConfig"
) -> None:
    """Apply genes controlling movement behavior.

    Adjusts particle velocity through vectorized operations, accounting for
    genetic parameters, environmental friction, and stochastic factors.

    Args:
        particle: Cellular type data containing position and velocity vectors
        gene_data: Movement parameters with structure:
            [0]: speed_modifier - Base velocity multiplier (default: 1.0)
            [1]: randomness - Stochastic movement factor (default: 0.1)
            [2]: direction_bias - Directional preference (default: 0.0)
        env: Environmental configuration containing friction values

    Returns:
        None: Updates particle velocity and energy in-place
    """
    # Extract gene parameters with defaults for missing values using safer indexing
    params = _extract_gene_parameters(
        gene_data,
        defaults=[1.0, 0.1, 0.0],
        bounds=[(0.1, 3.0), (0.0, 1.0), (-1.0, 1.0)],
    )

    speed_modifier, randomness, direction_bias = params

    # Apply environmental physics - convert friction to retention factor
    friction_factor = 1.0 - env.friction

    # Generate stochastic component for movement variation
    stochastic_x = randomness * np.random.uniform(-1, 1, size=particle.vx.size)
    stochastic_y = randomness * np.random.uniform(-1, 1, size=particle.vy.size)

    # Update velocity vectors through vectorized operations
    particle.vx = (
        particle.vx * friction_factor * speed_modifier + stochastic_x + direction_bias
    )

    particle.vy = (
        particle.vy * friction_factor * speed_modifier + stochastic_y + direction_bias
    )

    # Calculate energy expenditure proportional to movement magnitude
    velocity_magnitude = np.sqrt(np.power(particle.vx, 2) + np.power(particle.vy, 2))
    energy_cost = (
        velocity_magnitude * 0.01 * speed_modifier
    )  # Higher speed = higher cost

    # Apply energy cost with vectorized minimum boundary check
    particle.energy = np.maximum(0.0, particle.energy - energy_cost)


def apply_interaction_gene(
    particle: "CellularTypeData",
    others: List["CellularTypeData"],
    gene_data: GeneData,
    env: "SimulationConfig",
) -> None:
    """Apply interaction-related behavior based on proximity.

    Calculates inter-particle forces using vectorized operations to simulate
    social behaviors like flocking, avoidance, or cooperation.

    Args:
        particle: Cellular type data of the active particles
        others: List of other cellular types for interaction calculations
        gene_data: Interaction parameters with structure:
            [0]: attraction_strength - Force magnitude (+attract, -repel) (default: 0.5)
            [1]: interaction_radius - Maximum interaction distance (default: 100.0)
        env: Environmental configuration parameters

    Returns:
        None: Updates particle velocity and energy in-place
    """
    # Extract interaction parameters with bounds validation
    params = _extract_gene_parameters(
        gene_data, defaults=[0.5, 100.0], bounds=[(-2.0, 2.0), (10.0, 300.0)]
    )

    attraction_strength: float = params[0]
    interaction_radius: float = params[1]

    # Apply species-specific interaction modifiers if specified in environment
    species_idx = (
        particle.species_id[0]
        if hasattr(particle, "species_id") and len(particle.species_id) > 0
        else None
    )

    # Type-safe attribute check and interaction matrix access
    if (
        hasattr(env, "species_interaction_matrix")
        and species_idx is not None
        and getattr(env, "species_interaction_matrix", None) is not None
        and species_idx < len(getattr(env, "species_interaction_matrix"))
    ):
        # Apply species-specific modifier
        species_matrix = cast(List[float], getattr(env, "species_interaction_matrix"))
        # Explicitly cast the matrix element to float before conversion
        matrix_value = float(cast(float, species_matrix[species_idx]))
        attraction_strength = float(attraction_strength * matrix_value)

    # Process each potential interaction target
    for other in others:
        if other == particle:
            continue  # Skip self-interaction

        # Calculate vectorized distance matrix between all particle pairs
        dx: FloatArray = (
            other.x - particle.x[:, np.newaxis]
        )  # Broadcasting for all combinations
        dy: FloatArray = other.y - particle.y[:, np.newaxis]
        distances: FloatArray = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

        # Create interaction mask for distance-based filtering
        interact_mask: BoolArray = (distances > 0.0) & (distances < interaction_radius)

        if not np.any(interact_mask):
            continue  # Skip if no particles are within interaction range

        # Calculate normalized direction vectors with safe division
        with np.errstate(divide="ignore", invalid="ignore"):
            dx_norm: FloatArray = np.where(distances > 0, dx / distances, 0)
            dy_norm: FloatArray = np.where(distances > 0, dy / distances, 0)
        # Calculate interaction force with distance-based falloff
        force_magnitudes: FloatArray = attraction_strength * (
            1.0 - distances / interaction_radius
        )

        # Create typed intermediate variables for clarity
        dx_force: FloatArray = (dx_norm * force_magnitudes * interact_mask).astype(
            np.float64
        )
        dy_force: FloatArray = (dy_norm * force_magnitudes * interact_mask).astype(
            np.float64
        )

        # Apply forces to update velocity vectors with explicit types
        particle.vx += np.sum(dx_force, axis=1)
        particle.vy += np.sum(dy_force, axis=1)

        # Apply energy cost proportional to interaction count
        # More interactions = higher communication/coordination cost
        interaction_count = np.sum(interact_mask, axis=1)
        energy_cost = 0.01 * interaction_count
        particle.energy = np.maximum(0.0, particle.energy - energy_cost)


def apply_energy_gene(
    particle: "CellularTypeData", gene_data: GeneData, env: "SimulationConfig"
) -> None:
    """Regulate energy dynamics based on genetic and environmental factors.

    Simulates metabolic processes including passive energy gain, efficiency,
    and age-based decay through vectorized operations.

    Args:
        particle: Cellular type data containing energy attributes
        gene_data: Energy parameters with structure:
            [0]: passive_gain - Base energy acquisition rate (default: 0.1)
            [1]: feeding_efficiency - Nutrient absorption rate (default: 0.5)
            [2]: predation_efficiency - Energy extraction from prey (default: 0.3)
        env: Environmental configuration parameters

    Returns:
        None: Updates particle energy levels in-place
    """
    # Extract energy parameters with proper bounds
    params = _extract_gene_parameters(
        gene_data, defaults=[0.1, 0.5, 0.3], bounds=[(0.0, 0.5), (0.1, 1.0), (0.1, 1.0)]
    )

    passive_gain, feeding_efficiency, predation_efficiency = params

    # Update predation_efficiency attribute if it exists
    if hasattr(particle, "predation_efficiency"):
        # Update the vector of predation efficiency values
        particle.predation_efficiency = np.full_like(
            particle.predation_efficiency, predation_efficiency
        )

    # Environmental modifiers affecting energy dynamics
    env_modifier = _calculate_environmental_modifier(particle, env)

    # Calculate base energy acquisition
    base_gain = passive_gain * particle.energy_efficiency
    energy_gain = base_gain * env_modifier * feeding_efficiency

    # Apply energy gain vectorized
    particle.energy += energy_gain

    # Apply age-based energy decay (senescence)
    age_factor = np.clip(particle.age / particle.max_age, 0.0, 1.0)
    energy_decay = 0.01 * age_factor * (1.0 + age_factor)  # Quadratic age penalty

    # Apply decay and enforce energy bounds
    particle.energy = np.maximum(0.0, particle.energy - energy_decay)
    particle.energy = np.minimum(particle.energy, particle.max_energy)

    # Update alive status based on energy level
    particle.alive = particle.energy > 0.0


def apply_growth_gene(
    particle: "CellularTypeData", gene_data: GeneData, env: "SimulationConfig"
) -> None:
    """Apply growth gene effects to energy and physical attributes.

    Controls developmental processes including energy utilization,
    size changes, and maturation based on genetic factors.

    Args:
        particle: Cellular type data to apply growth modifications to
        gene_data: Growth parameters with structure:
            [0]: growth_rate - Base development speed (default: 0.1)
            [1]: adult_size - Target mature size (default: 1.0)
            [2]: maturity_age - Age at which growth stabilizes (default: 50.0)
        env: Environmental configuration parameters

    Returns:
        None: Updates particle attributes in-place
    """
    # Extract growth parameters
    params = _extract_gene_parameters(
        gene_data,
        defaults=[0.1, 1.0, 50.0],
        bounds=[(0.05, 0.5), (0.5, 2.0), (20.0, 150.0)],
    )

    growth_rate, adult_size, maturity_age = params

    # Calculate developmental stage - sigmoid function for smooth transition
    maturity_factor = 1.0 / (1.0 + np.exp(-(particle.age - maturity_age) / 10.0))

    # Adjust growth rate based on developmental stage
    effective_growth = growth_rate * (1.0 - maturity_factor) + 1.0 * maturity_factor

    # Calculate size factor - affects energy bounds and other physical attributes
    size_factor = (
        1.0 - maturity_factor
    ) * effective_growth + maturity_factor * adult_size

    # Apply energy scaling within physiological bounds
    min_energy = particle.min_energy
    max_energy = particle.max_energy * size_factor

    # Apply growth effects to energy with bounds enforcement
    particle.energy = np.clip(
        particle.energy * effective_growth, min_energy, max_energy
    )

    # Update physical attributes if size-dependent
    if particle.mass_based and particle.mass is not None:
        # Get base mass with safe fallback to current mass
        base_mass = getattr(particle, "base_mass", particle.mass.copy())
        # Scale mass based on developmental stage and adult size target
        particle.mass = base_mass * size_factor


def apply_reproduction_gene(
    particle: "CellularTypeData",
    others: List["CellularTypeData"],
    gene_data: GeneData,
    env: "SimulationConfig",
) -> None:
    """Handle sexual and asexual reproduction mechanics.

    Controls particle reproduction based on energy thresholds, creates
    offspring with inherited traits and mutations, and manages speciation
    through genetic distance calculations.

    Args:
        particle: Cellular type data of the potential parent particles
        others: List of other cellular types for potential sexual reproduction
        gene_data: Reproduction parameters with structure:
            [0]: sexual_threshold - Energy required for sexual reproduction (default: 150.0)
            [1]: asexual_threshold - Energy required for asexual reproduction (default: 100.0)
            [2]: reproduction_cost - Energy expended per reproduction (default: 50.0)
            [3]: cooldown_time - Minimum age between reproduction events (default: 30.0)
        env: Environmental configuration containing genetics parameters

    Returns:
        None: Creates new particles through side effects
    """
    # Extract reproduction parameters with appropriate bounds
    params = _extract_gene_parameters(
        gene_data,
        defaults=[150.0, 100.0, 50.0, 30.0],
        bounds=[(100.0, 200.0), (50.0, 150.0), (25.0, 100.0), (10.0, 100.0)],
    )

    sexual_threshold, asexual_threshold, reproduction_cost, cooldown_time = params

    # Sexual reproduction not implemented in this function but threshold preserved
    # for potential future extension to sexual reproduction mechanics
    _ = sexual_threshold

    # Check reproduction conditions (energy > threshold, mature age, alive)
    can_reproduce = (
        (particle.energy > asexual_threshold)
        & (particle.age > cooldown_time)
        & particle.alive
    )

    if not np.any(can_reproduce):
        return  # No particles ready to reproduce

    # Get indices of particles that can reproduce
    reproduce_indices = np.where(can_reproduce)[0]

    # Define trait parameters for mutation and inheritance
    trait_params = _get_trait_mutation_parameters(env.genetics)

    for idx in reproduce_indices:
        # Deduct energy cost from parent
        particle.energy[idx] -= reproduction_cost

        # Get mutation probability and range from environment
        mutation_rate = env.genetics.gene_mutation_rate
        mutation_range = env.genetics.gene_mutation_range

        # Generate offspring traits through mutation
        offspring_traits = _generate_offspring_traits(
            particle, idx, mutation_rate, mutation_range, trait_params
        )

        # Calculate genetic distance for speciation
        genetic_distance = _calculate_genetic_distance(particle, idx, offspring_traits)

        # Determine species ID based on genetic distance threshold
        species_id_val = _determine_species_id(
            particle, idx, genetic_distance, env.speciation_threshold
        )

        # Add offspring to particle population
        _add_offspring_to_population(
            particle, idx, offspring_traits, species_id_val, reproduction_cost
        )


def apply_predation_gene(
    particle: "CellularTypeData",
    others: List["CellularTypeData"],
    gene_data: GeneData,
    env: "SimulationConfig",
) -> None:
    """Apply predation behaviors based on genetic predatory traits.

    Controls predatory interactions between particles including target
    selection, attack success probability, and energy transfer mechanics.

    Args:
        particle: Cellular type data of the predator particle
        others: List of other cellular types for potential predation targets
        gene_data: Predation parameters with structure:
            [0]: attack_power - Base attack strength (default: 1.0)
            [1]: energy_conversion - Efficiency of converting prey to energy (default: 0.5)
            [2]: predation_strategy - Hunting behavior selector (default: 0.0)
            [3]: detection_range - Maximum distance to detect prey (default: 100.0)
        env: Environmental configuration parameters

    Returns:
        None: Updates predator energy and potentially removes prey particles
    """
    # Extract predation parameters
    params = _extract_gene_parameters(
        gene_data,
        defaults=[1.0, 0.5, 0.0, 100.0],
        bounds=[(0.1, 5.0), (0.1, 0.9), (0.0, 3.0), (20.0, 300.0)],
    )

    attack_power, energy_conversion, strategy_selector, detection_range = params

    # Determine predation strategy based on selector value
    strategy = _select_predation_strategy(strategy_selector)

    # Only living predators with sufficient energy can hunt
    active_predators = (particle.energy > 10.0) & particle.alive

    if not np.any(active_predators):
        return  # No predators are capable of hunting

    predator_indices = np.where(active_predators)[0]

    for other in others:
        if other == particle or not np.any(other.alive):
            continue  # Skip self or groups with no living prey

        for pred_idx in predator_indices:
            # Skip if this predator has already consumed prey this cycle
            if particle.energy[pred_idx] > particle.max_energy * 0.9:
                continue

            # Calculate distances to all potential prey
            distances = np.sqrt(
                np.power(particle.x[pred_idx] - other.x, 2)
                + np.power(particle.y[pred_idx] - other.y, 2)
            )

            # Identify valid prey within detection range
            valid_prey = (distances < detection_range) & other.alive

            if not np.any(valid_prey):
                continue  # No valid prey within range

            # Select prey based on strategy
            prey_idx = _select_prey(
                other, valid_prey, strategy, distances, particle, pred_idx
            )

            if prey_idx is None:
                continue  # No suitable prey found

            # Determine attack success probability
            success_prob = _calculate_attack_success(
                predator=particle,
                pred_idx=pred_idx,
                prey=other,
                prey_idx=prey_idx,
                attack_power=attack_power,
            )

            # Execute attack if successful
            if np.random.random() < success_prob:
                # Transfer energy from prey to predator
                energy_gained = other.energy[prey_idx] * energy_conversion
                particle.energy[pred_idx] += energy_gained

                # Cap predator energy at maximum
                particle.energy[pred_idx] = min(
                    float(particle.energy[pred_idx]), particle.max_energy
                )

                # Prey loses all energy (dies)
                other.energy[prey_idx] = 0.0
                other.alive[prey_idx] = False

                # Apply attack cooldown to predator if the attribute exists
                if hasattr(particle, "cooldown"):
                    particle.cooldown[pred_idx] = 10.0  # Predator needs recovery time

                # Break the inner loop - one successful attack per predator per cycle
                break


# Helper functions for gene application


def _extract_gene_parameters(
    gene_data: GeneData, defaults: List[float], bounds: List[Tuple[float, float]]
) -> List[float]:
    """Extract gene parameters with defaults and bounds enforcement.

    Args:
        gene_data: Raw gene data array
        defaults: Default values for each parameter
        bounds: Min/max bounds for each parameter as (min, max) tuples

    Returns:
        List of parsed and bounded parameter values
    """
    result: List[float] = []
    for i, (default, (min_val, max_val)) in enumerate(zip(defaults, bounds)):
        # Extract value with default fallback
        value = gene_data[i] if i < len(gene_data) else default
        # Enforce bounds
        value = np.clip(value, min_val, max_val)
        result.append(value)

    return result


def _calculate_environmental_modifier(
    particle: "CellularTypeData", env: "SimulationConfig"
) -> float:
    """Calculate environmental modifiers for energy dynamics.

    Args:
        particle: Particle data
        env: Environmental configuration

    Returns:
        Environmental modifier multiplier
    """
    # Start with baseline modifier
    modifier: float = 1.0

    # Apply day/night cycle effect if configured
    if hasattr(env, "day_night_cycle") and getattr(env, "day_night_cycle", False):
        # Safely access time and day_length with defaults
        env_time = getattr(env, "time", 0.0)
        day_length = getattr(env, "day_length", 24.0)

        # Day/night cycle affects energy production (e.g., photosynthesis)
        cycle_factor = np.sin(env_time / day_length * 2 * np.pi) * 0.5 + 0.5
        modifier *= 0.5 + cycle_factor

    # Apply temperature effects if configured
    if hasattr(env, "temperature"):
        # Temperature affects metabolic rates
        temp_optimal = 0.5  # Normalized optimal temperature
        env_temp = getattr(env, "temperature", temp_optimal)
        temp_factor = 1.0 - abs(env_temp - temp_optimal) * 2
        modifier *= max(0.1, temp_factor)

    return modifier


def _get_trait_mutation_parameters(genetics: "GeneticParamConfig") -> Dict[str, float]:
    """Define mutation parameters for all traits.

    Args:
        genetics: Genetics configuration containing mutation settings

    Returns:
        Dictionary of trait names to their maximum allowed values
    """
    return {
        "energy_efficiency": 1.0,
        "speed_factor": 2.0,
        "interaction_strength": 2.0,
        "perception_range": 300.0,
        "reproduction_rate": 1.0,
        "synergy_affinity": 1.0,
        "colony_factor": 1.0,
        "drift_sensitivity": 1.0,
    }


def _generate_offspring_traits(
    particle: "CellularTypeData",
    idx: int,
    mutation_rate: float,
    mutation_range: Tuple[float, float],
    trait_params: Dict[str, float],
) -> Dict[str, FloatArray]:
    """Generate offspring traits through mutation of parent traits.

    Args:
        particle: Parent particle data
        idx: Index of the parent particle
        mutation_rate: Probability of mutation per trait
        mutation_range: (min, max) range for mutation magnitude
        trait_params: Maximum values for each trait

    Returns:
        Dictionary of trait names to their mutated values
    """
    offspring_traits: Dict[str, FloatArray] = {}

    # For each trait, apply mutation based on parent's value
    for trait_name, max_value in trait_params.items():
        # Get parent trait value
        parent_value = getattr(particle, trait_name)[idx]

        # Create single-element arrays for the mutation function
        parent_array = np.array([parent_value])
        mutate_flag = np.array([mutation_rate > np.random.random()], dtype=bool)

        # Apply mutation
        offspring_traits[trait_name] = mutate_trait(
            parent_array, mutate_flag, mutation_range[0], max_value
        )

    return offspring_traits


def _calculate_genetic_distance(
    particle: "CellularTypeData", idx: int, offspring_traits: Dict[str, FloatArray]
) -> float:
    """Calculate genetic distance between parent and offspring.

    Args:
        particle: Parent particle data
        idx: Index of the parent particle
        offspring_traits: Dictionary of offspring trait values

    Returns:
        Normalized genetic distance as a float
    """
    # Calculate squared differences for each trait
    squared_diffs: List[float] = []
    traits_to_compare = [
        "speed_factor",
        "interaction_strength",
        "perception_range",
        "reproduction_rate",
        "synergy_affinity",
        "colony_factor",
        "drift_sensitivity",
    ]

    for trait in traits_to_compare:
        parent_val = getattr(particle, trait)[idx]
        offspring_val = float(offspring_traits[trait][0])

        # Get the max value for this trait for normalization
        max_val = 1.0
        if trait == "perception_range":
            max_val = 300.0
        elif trait in ["speed_factor", "interaction_strength"]:
            max_val = 2.0

        # Calculate normalized squared difference
        normalized_diff = ((offspring_val - parent_val) / max_val) ** 2
        squared_diffs.append(normalized_diff)

    # Calculate Euclidean distance across normalized trait space
    genetic_distance = np.sqrt(np.sum(squared_diffs))
    return float(genetic_distance)


def _determine_species_id(
    particle: "CellularTypeData",
    idx: int,
    genetic_distance: float,
    speciation_threshold: float,
) -> int:
    """Determine species ID based on genetic distance.

    Args:
        particle: Parent particle data
        idx: Index of the parent particle
        genetic_distance: Calculated genetic distance
        speciation_threshold: Threshold beyond which speciation occurs

    Returns:
        Species ID as an integer
    """
    # Check if speciation should occur based on genetic distance
    if genetic_distance > speciation_threshold:
        # Speciation event - create new species
        # Safely get the maximum species_id with type checking
        if hasattr(particle, "species_id") and len(particle.species_id) > 0:
            # Create a new species with ID one higher than the current maximum
            current_max = np.max(particle.species_id)
            species_id_val = int(current_max) + 1
        else:
            # Fallback if species_id is missing or empty
            species_id_val = 1
    else:
        # Same species as parent - use parent's species ID with safe access
        species_id_val = (
            int(particle.species_id[idx]) if hasattr(particle, "species_id") else 0
        )

    return species_id_val


def _add_offspring_to_population(
    particle: "CellularTypeData",
    idx: int,
    offspring_traits: Dict[str, FloatArray],
    species_id_val: int,
    reproduction_cost: float,
) -> None:
    """Add new offspring to the particle population.

    Args:
        particle: Parent particle data
        idx: Index of the parent particle
        offspring_traits: Dictionary of offspring trait values
        species_id_val: Determined species ID
        reproduction_cost: Energy cost of reproduction

    Returns:
        None: Modifies particle population in-place
    """
    # Calculate initial position with small random offset
    pos_x = float(particle.x[idx] + np.random.uniform(-5, 5))
    pos_y = float(particle.y[idx] + np.random.uniform(-5, 5))

    # Calculate initial velocity with small random variation
    vel_x = float(particle.vx[idx] * np.random.uniform(0.9, 1.1))
    vel_y = float(particle.vy[idx] * np.random.uniform(0.9, 1.1))

    # Calculate initial energy (half of parent's energy)
    initial_energy = float(particle.energy[idx] * 0.5)

    # Get mass value if applicable
    mass_val = None
    if particle.mass_based and particle.mass is not None:
        mass_val = float(particle.mass[idx])

    # Add new particle to the population
    particle.add_component(
        x=pos_x,
        y=pos_y,
        vx=vel_x,
        vy=vel_y,
        energy=initial_energy,
        mass_val=mass_val,
        # Extract scalar float values from numpy arrays
        energy_efficiency_val=float(offspring_traits["energy_efficiency"][0]),
        speed_factor_val=float(offspring_traits["speed_factor"][0]),
        interaction_strength_val=float(offspring_traits["interaction_strength"][0]),
        perception_range_val=float(offspring_traits["perception_range"][0]),
        reproduction_rate_val=float(offspring_traits["reproduction_rate"][0]),
        synergy_affinity_val=float(offspring_traits["synergy_affinity"][0]),
        colony_factor_val=float(offspring_traits["colony_factor"][0]),
        drift_sensitivity_val=float(offspring_traits["drift_sensitivity"][0]),
        species_id_val=species_id_val,
        parent_id_val=int(particle.type_id),
        max_age=float(particle.max_age),
    )


def _select_predation_strategy(strategy_selector: float) -> PredationStrategy:
    """Select predation strategy based on genetic selector value.

    Args:
        strategy_selector: Numeric value determining strategy

    Returns:
        PredationStrategy enum value
    """
    if strategy_selector < 0.75:
        return PredationStrategy.OPPORTUNISTIC
    elif strategy_selector < 1.5:
        return PredationStrategy.ENERGY_OPTIMAL
    elif strategy_selector < 2.25:
        return PredationStrategy.SIZE_BASED
    else:
        return PredationStrategy.TERRITORIAL


def _select_prey(
    prey: "CellularTypeData",
    valid_mask: BoolArray,
    strategy: PredationStrategy,
    distances: FloatArray,
    predator: "CellularTypeData",
    pred_idx: int,
) -> Optional[int]:
    """Select optimal prey based on predation strategy.

    Args:
        prey: Potential prey particle data
        valid_mask: Boolean mask of valid prey candidates
        strategy: Selected predation strategy
        distances: Distances to each potential prey
        predator: Predator particle data
        pred_idx: Index of predator particle

    Returns:
        Selected prey index or None if no suitable prey found
    """
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return None

    if strategy == PredationStrategy.OPPORTUNISTIC:
        # Choose closest prey
        closest_idx = np.argmin(distances[valid_mask])
        return valid_indices[closest_idx]

    elif strategy == PredationStrategy.ENERGY_OPTIMAL:
        # Choose prey with highest energy
        energy_values = prey.energy[valid_mask]
        if np.all(energy_values <= 0):
            return None
        best_idx = np.argmax(energy_values)
        return valid_indices[best_idx]

    elif strategy == PredationStrategy.SIZE_BASED:
        # Choose smallest prey (if size/mass is tracked)
        if prey.mass_based and prey.mass is not None:
            mass_values = prey.mass[valid_mask]
            smallest_idx = np.argmin(mass_values)
            return valid_indices[smallest_idx]
        else:
            # Fall back to energy as size proxy
            energy_values = prey.energy[valid_mask]
            smallest_idx = np.argmin(energy_values)
            return valid_indices[smallest_idx]

    elif strategy == PredationStrategy.TERRITORIAL:
        # Choose prey closest to predator's territory center
        # For simplicity, use current position as territory center
        center_x, center_y = predator.x[pred_idx], predator.y[pred_idx]
        territory_distances = np.sqrt(
            np.power(prey.x[valid_mask] - center_x, 2)
            + np.power(prey.y[valid_mask] - center_y, 2)
        )
        closest_idx = np.argmin(territory_distances)
        return valid_indices[closest_idx]

    # Default fallback - choose random valid prey
    return int(np.random.choice(valid_indices))


def _calculate_attack_success(
    predator: "CellularTypeData",
    pred_idx: int,
    prey: "CellularTypeData",
    prey_idx: int,
    attack_power: float,
) -> float:
    """Calculate probability of successful predation.

    Args:
        predator: Predator particle data
        pred_idx: Index of predator particle
        prey: Prey particle data
        prey_idx: Index of prey particle
        attack_power: Base attack strength

    Returns:
        Probability of successful attack (0.0-1.0)
    """
    # Base success rate determined by attack power
    base_success = min(0.9, attack_power * 0.2)

    # Energy ratio factor - predators with more energy relative to prey have advantage
    energy_ratio = predator.energy[pred_idx] / max(1.0, prey.energy[prey_idx])
    energy_factor = min(2.0, max(0.5, energy_ratio))

    # Size/mass advantage if applicable
    size_factor = 1.0
    if (
        predator.mass_based
        and prey.mass_based
        and predator.mass is not None
        and prey.mass is not None
    ):
        mass_ratio = predator.mass[pred_idx] / max(0.1, prey.mass[prey_idx])
        size_factor = min(2.0, max(0.2, mass_ratio))

    # Calculate final success probability
    success_prob = base_success * energy_factor * size_factor

    # Cap at reasonable bounds
    return min(0.95, max(0.05, success_prob))
