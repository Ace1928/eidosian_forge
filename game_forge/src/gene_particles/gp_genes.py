from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

import numpy as np
from eidosian_core import eidosian

# Use TYPE_CHECKING for circular imports
if TYPE_CHECKING:
    from game_forge.src.gene_particles.gp_config import (
        GeneticParamConfig,
        SimulationConfig,
    )
    from game_forge.src.gene_particles.gp_types import CellularTypeData

from game_forge.src.gene_particles.gp_types import BoolArray, FloatArray, GeneData, IntArray


# Import scipy's KDTree with fallback for environments without scipy
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore[import]
except ImportError:
    class KDTree:
        """Fallback KDTree with query_ball_point stub for headless tests."""

        def __init__(self, data: FloatArray, leafsize: int = 10) -> None:
            _ = data, leafsize

        @eidosian()
        def query_ball_point(
            self, x: FloatArray, r: float, p: float = 2.0, eps: float = 0.0
        ) -> List[List[int]]:
            _ = x, r, p, eps
            return [[] for _ in range(len(x))]
from game_forge.src.gene_particles.gp_utility import (
    mutate_trait,
    tile_positions_for_wrap,
    wrap_deltas,
    wrap_positions,
)


class PredationStrategy(Enum):
    """Predation strategies used by cellular entities."""

    OPPORTUNISTIC = auto()  # Attack when opportunity arises
    ENERGY_OPTIMAL = auto()  # Attack only when energy benefit exceeds cost
    SIZE_BASED = auto()  # Attack smaller entities preferentially
    TERRITORIAL = auto()  # Attack entities in proximity to territory


@eidosian()
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
    if env.spatial_dimensions == 3:
        stochastic_z = randomness * np.random.uniform(-1, 1, size=particle.vz.size)

    # Update velocity vectors through vectorized operations
    particle.vx = (
        particle.vx * friction_factor * speed_modifier + stochastic_x + direction_bias
    )

    particle.vy = (
        particle.vy * friction_factor * speed_modifier + stochastic_y + direction_bias
    )
    if env.spatial_dimensions == 3:
        particle.vz = (
            particle.vz * friction_factor * speed_modifier + stochastic_z + direction_bias
        )

    # Calculate energy expenditure proportional to movement magnitude
    if env.spatial_dimensions == 3:
        velocity_magnitude = np.sqrt(
            np.power(particle.vx, 2)
            + np.power(particle.vy, 2)
            + np.power(particle.vz, 2)
        )
    else:
        velocity_magnitude = np.sqrt(
            np.power(particle.vx, 2) + np.power(particle.vy, 2)
        )
    energy_cost = (
        velocity_magnitude * 0.01 * speed_modifier
    )  # Higher speed = higher cost

    # Apply energy cost with vectorized minimum boundary check
    particle.energy = np.maximum(0.0, particle.energy - energy_cost)


@eidosian()
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
        if env.boundary_mode == "wrap":
            world_width = getattr(env, "world_width", None)
            world_height = getattr(env, "world_height", None)
            if world_width:
                dx = wrap_deltas(dx, float(world_width))
            if world_height:
                dy = wrap_deltas(dy, float(world_height))
        if env.spatial_dimensions == 3:
            dz: FloatArray = other.z - particle.z[:, np.newaxis]
            if env.boundary_mode == "wrap":
                world_depth = getattr(env, "world_depth", None)
                if world_depth:
                    dz = wrap_deltas(dz, float(world_depth))
            distances: FloatArray = np.sqrt(
                np.power(dx, 2) + np.power(dy, 2) + np.power(dz, 2)
            )
        else:
            distances = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

        # Create interaction mask for distance-based filtering
        interact_mask: BoolArray = (distances > 0.0) & (distances < interaction_radius)

        if not np.any(interact_mask):
            continue  # Skip if no particles are within interaction range

        # Calculate normalized direction vectors with safe division
        with np.errstate(divide="ignore", invalid="ignore"):
            dx_norm: FloatArray = np.where(distances > 0, dx / distances, 0)
            dy_norm: FloatArray = np.where(distances > 0, dy / distances, 0)
            if env.spatial_dimensions == 3:
                dz_norm: FloatArray = np.where(distances > 0, dz / distances, 0)
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
        if env.spatial_dimensions == 3:
            dz_force: FloatArray = (
                dz_norm * force_magnitudes * interact_mask
            ).astype(np.float64)

        # Apply forces to update velocity vectors with explicit types
        particle.vx += np.sum(dx_force, axis=1)
        particle.vy += np.sum(dy_force, axis=1)
        if env.spatial_dimensions == 3:
            particle.vz += np.sum(dz_force, axis=1)

        # Apply energy cost proportional to interaction count
        # More interactions = higher communication/coordination cost
        interaction_count = np.sum(interact_mask, axis=1)
        energy_cost = 0.01 * interaction_count
        particle.energy = np.maximum(0.0, particle.energy - energy_cost)


@eidosian()
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


@eidosian()
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


@eidosian()
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
            [4]: mate_radius_multiplier - Scales mate selection radius (default: 1.0)
            [5]: sexual_bias - Probability multiplier for sexual reproduction (default: 0.65)
            [6]: mutation_boost - Multiplier for mutation rate during sexual reproduction (default: 1.0)
        env: Environmental configuration containing genetics parameters

    Returns:
        None: Creates new particles through side effects
    """
    # Extract reproduction parameters with appropriate bounds
    params = _extract_gene_parameters(
        gene_data,
        defaults=[150.0, 100.0, 50.0, 30.0, 1.0, 0.65, 1.0],
        bounds=[
            (80.0, 300.0),
            (50.0, 200.0),
            (10.0, 150.0),
            (5.0, 200.0),
            (0.25, 3.0),
            (0.0, 1.0),
            (0.5, 2.0),
        ],
    )

    (
        sexual_threshold,
        asexual_threshold,
        reproduction_cost,
        cooldown_time,
        mate_radius_multiplier,
        sexual_bias,
        mutation_boost,
    ) = params

    if not env.sexual_reproduction_enabled and not env.asexual_reproduction_enabled:
        return

    base_count = particle.x.size
    eligible_base = particle.alive[:base_count] & (
        particle.age[:base_count] > cooldown_time
    )
    if not np.any(eligible_base):
        return

    # Define trait ranges for mutation and inheritance
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    used_mask = np.zeros(particle.x.size, dtype=bool)

    # Sexual reproduction pathway
    if env.sexual_reproduction_enabled:
        sexual_candidates = eligible_base & (
            particle.energy[:base_count] > sexual_threshold
        )
        if np.any(sexual_candidates):
            mate_radius = env.mate_selection_radius * mate_radius_multiplier
            parent_a_idx, parent_b_idx = _select_mating_pairs(
                particle,
                sexual_candidates,
                env,
                trait_ranges,
                mate_radius,
            )

            if parent_a_idx.size > 0:
                available_slots = max(0, env.max_particles_per_type - particle.x.size)
                if available_slots <= 0:
                    return

                if parent_a_idx.size > available_slots:
                    chosen = np.random.choice(
                        parent_a_idx.size, size=available_slots, replace=False
                    )
                    parent_a_idx = parent_a_idx[chosen]
                    parent_b_idx = parent_b_idx[chosen]

                sexual_probability = float(
                    np.clip(
                        env.sexual_reproduction_probability * sexual_bias, 0.0, 1.0
                    )
                )
                select_mask = np.random.random(parent_a_idx.size) < sexual_probability
                parent_a_idx = parent_a_idx[select_mask]
                parent_b_idx = parent_b_idx[select_mask]

                if parent_a_idx.size > 0:
                    used_mask[parent_a_idx] = True
                    used_mask[parent_b_idx] = True

                    offspring_traits, offspring_energy, species_ids = (
                        _produce_sexual_offspring(
                            particle,
                            parent_a_idx,
                            parent_b_idx,
                            env,
                            trait_ranges,
                            reproduction_cost,
                            mutation_boost,
                            others,
                        )
                    )

                    if offspring_energy.size > 0:
                        _append_offspring_bulk(
                            particle,
                            parent_a_idx,
                            offspring_traits,
                            species_ids,
                            offspring_energy,
                            mate_indices=parent_b_idx,
                        )

    # Asexual reproduction pathway
    if env.asexual_reproduction_enabled:
        asexual_candidates = eligible_base & (
            particle.energy[:base_count] > asexual_threshold
        )
        if np.any(asexual_candidates):
            asexual_candidates &= ~used_mask
            if not np.any(asexual_candidates):
                return  # pragma: no cover

            parent_indices = np.where(asexual_candidates)[0]
            available_slots = max(0, env.max_particles_per_type - particle.x.size)
            if available_slots <= 0:
                return

            if parent_indices.size > available_slots:
                parent_indices = np.random.choice(
                    parent_indices, size=available_slots, replace=False
                )

            offspring_traits, offspring_energy, species_ids = (
                _produce_asexual_offspring(
                    particle,
                    parent_indices,
                    env,
                    trait_ranges,
                    reproduction_cost,
                )
            )

            if offspring_energy.size > 0:
                _append_offspring_bulk(
                    particle,
                    parent_indices,
                    offspring_traits,
                    species_ids,
                    offspring_energy,
                )


@eidosian()
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
            dx = particle.x[pred_idx] - other.x
            dy = particle.y[pred_idx] - other.y
            if env.boundary_mode == "wrap":
                world_width = getattr(env, "world_width", None)
                world_height = getattr(env, "world_height", None)
                if world_width:
                    dx = wrap_deltas(dx, float(world_width))
                if world_height:
                    dy = wrap_deltas(dy, float(world_height))

            if env.spatial_dimensions == 3:
                dz = particle.z[pred_idx] - other.z
                if env.boundary_mode == "wrap":
                    world_depth = getattr(env, "world_depth", None)
                    if world_depth:
                        dz = wrap_deltas(dz, float(world_depth))
                distances = np.sqrt(np.power(dx, 2) + np.power(dy, 2) + np.power(dz, 2))
            else:
                distances = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

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


# Sexual reproduction helper functions


def _select_mating_pairs(
    particle: "CellularTypeData",
    candidate_mask: BoolArray,
    env: "SimulationConfig",
    trait_ranges: Dict[str, Tuple[float, float]],
    mate_radius: float,
) -> Tuple[IntArray, IntArray]:
    """Select mating pairs within a cellular type based on strategy and compatibility."""
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size < 2:
        return (
            np.array([], dtype=np.int_),
            np.array([], dtype=np.int_),
        )

    if env.spatial_dimensions == 3:
        positions = np.column_stack((particle.x, particle.y, particle.z))
    else:
        positions = np.column_stack((particle.x, particle.y))

    if env.boundary_mode == "wrap" and env.world_width and env.world_height:
        if env.spatial_dimensions == 3 and env.world_depth:
            world_size = (float(env.world_width), float(env.world_height), float(env.world_depth))
        else:
            world_size = (float(env.world_width), float(env.world_height))
        index_map = None
        try:
            positions = positions.copy()
            positions[:, 0] = wrap_positions(positions[:, 0], 0.0, world_size[0])
            positions[:, 1] = wrap_positions(positions[:, 1], 0.0, world_size[1])
            if env.spatial_dimensions == 3:
                positions[:, 2] = wrap_positions(positions[:, 2], 0.0, world_size[2])
            tree = KDTree(positions, boxsize=world_size)
            raw_neighbors = tree.query_ball_point(
                positions[candidate_indices], mate_radius
            )
        except TypeError:
            tiled_positions, index_map = tile_positions_for_wrap(positions, world_size)
            tree = KDTree(tiled_positions)
            raw_neighbors = tree.query_ball_point(
                positions[candidate_indices], mate_radius
            )
        neighbors_list = []
        for local_neighbors in raw_neighbors:
            if not local_neighbors:
                neighbors_list.append([])
                continue
            if index_map is None:
                neighbors_list.append(list(local_neighbors))
                continue
            mapped = [int(index_map[idx]) for idx in local_neighbors]
            neighbors_list.append(list(dict.fromkeys(mapped)))
    else:
        tree = KDTree(positions)
        neighbors_list = tree.query_ball_point(positions[candidate_indices], mate_radius)

    used = np.zeros(particle.x.size, dtype=bool)
    parent_a: List[int] = []
    parent_b: List[int] = []

    for local_idx, neighbors in enumerate(neighbors_list):
        parent_idx = int(candidate_indices[local_idx])
        if used[parent_idx]:
            continue

        filtered = [
            int(idx)
            for idx in neighbors
            if idx != parent_idx and candidate_mask[idx] and not used[idx]
        ]

        if not filtered:
            continue

        if env.mate_selection_max_neighbors > 0 and len(filtered) > env.mate_selection_max_neighbors:
            filtered = list(
                np.random.choice(
                    filtered, size=env.mate_selection_max_neighbors, replace=False
                )
            )

        candidate_arr = np.array(filtered, dtype=np.int_)
        distances = _calculate_pairwise_distance(
            particle, parent_idx, particle, candidate_arr, trait_ranges, env
        )

        if not env.allow_cross_type_mating:
            same_species = (
                particle.species_id[candidate_arr]
                == particle.species_id[parent_idx]
            )
            if not np.any(same_species):
                continue
            candidate_arr = candidate_arr[same_species]
            distances = distances[same_species]

        compatible_mask = distances <= env.compatibility_threshold
        if not np.any(compatible_mask):
            continue

        candidate_arr = candidate_arr[compatible_mask]
        distances = distances[compatible_mask]

        mate_idx = _choose_mate_index(
            particle, candidate_arr, distances, env
        )
        if mate_idx is None:
            continue

        parent_a.append(parent_idx)
        parent_b.append(mate_idx)
        used[parent_idx] = True
        used[mate_idx] = True

    return (
        np.asarray(parent_a, dtype=np.int_),
        np.asarray(parent_b, dtype=np.int_),
    )


def _choose_mate_index(
    particle: "CellularTypeData",
    candidate_indices: IntArray,
    distances: FloatArray,
    env: "SimulationConfig",
) -> Optional[int]:
    """Select a mate index according to the configured strategy."""
    if candidate_indices.size == 0:
        return None

    strategy = env.mate_selection_strategy
    if strategy == "random":
        return int(np.random.choice(candidate_indices))

    energies = particle.energy[candidate_indices]
    if strategy == "energy":
        if np.all(energies <= 0.0):
            return None
        return int(candidate_indices[int(np.argmax(energies))])

    if strategy == "compatibility":
        return int(candidate_indices[int(np.argmin(distances))])

    # Hybrid strategy: combine compatibility and energy
    energy_min = float(np.min(energies))
    energy_max = float(np.max(energies))
    if energy_max > energy_min:
        energy_norm = (energies - energy_min) / (energy_max - energy_min)
    else:
        energy_norm = np.zeros_like(energies, dtype=np.float64)

    compatibility = np.exp(-distances)
    score = (env.compatibility_weight * compatibility) + (
        env.energy_weight * energy_norm
    )
    return int(candidate_indices[int(np.argmax(score))])


def _calculate_pairwise_distance(
    parent_particle: "CellularTypeData",
    parent_idx: int,
    mate_particle: "CellularTypeData",
    mate_indices: IntArray,
    trait_ranges: Dict[str, Tuple[float, float]],
    env: "SimulationConfig",
) -> FloatArray:
    """Compute normalized genetic distance between a parent and candidate mates."""
    squared = np.zeros(mate_indices.size, dtype=np.float64)
    for trait in env.genetics.gene_traits:
        parent_val = getattr(parent_particle, trait)[parent_idx]
        mate_vals = getattr(mate_particle, trait)[mate_indices]
        span = trait_ranges[trait][1] - trait_ranges[trait][0]
        if span <= 0.0:
            span = 1.0
        squared += ((mate_vals - parent_val) / span) ** 2
    return np.sqrt(squared)


def _build_linkage_indices(
    trait_names: List[str], linkage_groups: List[List[str]]
) -> List[List[int]]:
    """Map linkage group names to trait indices with fallbacks."""
    trait_to_idx = {name: idx for idx, name in enumerate(trait_names)}
    groups: List[List[int]] = []
    for group in linkage_groups:
        indices = [trait_to_idx[name] for name in group if name in trait_to_idx]
        if indices:
            groups.append(indices)

    covered = {idx for group in groups for idx in group}
    for idx in range(len(trait_names)):
        if idx not in covered:
            groups.append([idx])

    return groups


def _choose_crossover_modes(
    weights: Dict[str, float], count: int
) -> List[str]:
    """Choose crossover modes for each pair based on weighted distribution."""
    modes = list(weights.keys())
    weights_arr = np.array([weights[mode] for mode in modes], dtype=np.float64)
    total = float(np.sum(weights_arr))
    if total <= 0.0:
        weights_arr = np.full_like(weights_arr, 1.0)
    probs = weights_arr / float(np.sum(weights_arr))
    return list(np.random.choice(modes, size=count, p=probs))


def _crossover_traits(
    parent_a: FloatArray,
    parent_b: FloatArray,
    env: "SimulationConfig",
    trait_names: List[str],
) -> FloatArray:
    """Combine parent traits into offspring traits using weighted crossover modes."""
    n_traits, count = parent_a.shape
    offspring = np.empty_like(parent_a)
    modes = _choose_crossover_modes(env.crossover_mode_weights, count)
    linkage_indices = _build_linkage_indices(trait_names, env.linkage_groups)

    for mode in set(modes):
        indices = np.array([i for i, m in enumerate(modes) if m == mode], dtype=np.int_)

        if mode == "uniform":
            offspring[:, indices] = parent_b[:, indices]
            group_mask = (
                np.random.random((len(linkage_indices), indices.size))
                < env.recombination_rate
            )
            for g_idx, trait_idx in enumerate(linkage_indices):
                select = group_mask[g_idx]
                if not np.any(select):
                    continue
                offspring[np.ix_(trait_idx, indices[select])] = parent_a[
                    np.ix_(trait_idx, indices[select])
                ]

        elif mode == "arithmetic":
            weights = np.random.uniform(0.0, 1.0, size=(1, indices.size))
            offspring[:, indices] = (
                weights * parent_a[:, indices]
                + (1.0 - weights) * parent_b[:, indices]
            )

        elif mode == "blend":
            alpha = env.crossover_blend_alpha
            min_vals = np.minimum(parent_a[:, indices], parent_b[:, indices])
            max_vals = np.maximum(parent_a[:, indices], parent_b[:, indices])
            span = max_vals - min_vals
            lower = min_vals - alpha * span
            upper = max_vals + alpha * span
            offspring[:, indices] = np.random.uniform(lower, upper)

        else:  # segment
            offspring[:, indices] = parent_b[:, indices]
            for col_offset, pair_idx in enumerate(indices):
                if len(linkage_indices) == 1:
                    pivot = 1
                else:
                    pivot = int(np.random.randint(1, len(linkage_indices)))
                for group_idx, trait_idx in enumerate(linkage_indices):
                    if group_idx < pivot:
                        offspring[np.ix_(trait_idx, [pair_idx])] = parent_a[
                            np.ix_(trait_idx, [pair_idx])
                        ]

    if env.crossover_jitter > 0.0:
        offspring += np.random.normal(
            0.0, env.crossover_jitter, size=offspring.shape
        ).astype(np.float64)

    return offspring


def _collect_trait_matrix(
    particle: "CellularTypeData",
    indices: IntArray,
    trait_names: List[str],
) -> FloatArray:
    """Collect trait values as a stacked matrix for crossover."""
    return np.vstack([getattr(particle, trait)[indices] for trait in trait_names]).astype(
        np.float64
    )


def _produce_sexual_offspring(
    particle: "CellularTypeData",
    parent_a_idx: IntArray,
    parent_b_idx: IntArray,
    env: "SimulationConfig",
    trait_ranges: Dict[str, Tuple[float, float]],
    reproduction_cost: float,
    mutation_boost: float,
    others: List["CellularTypeData"],
) -> Tuple[Dict[str, FloatArray], FloatArray, IntArray]:
    """Generate sexual offspring traits and energy with crossover and mutation."""
    parent_a_energy = np.maximum(
        0.0, particle.energy[parent_a_idx] - reproduction_cost
    )
    parent_b_energy = np.maximum(
        0.0, particle.energy[parent_b_idx] - reproduction_cost
    )

    if env.allow_hybridization and env.hybridization_cost > 0.0:
        hybrid_mask = (
            particle.species_id[parent_a_idx] != particle.species_id[parent_b_idx]
        )
        if np.any(hybrid_mask):
            parent_a_energy[hybrid_mask] = np.maximum(
                0.0, parent_a_energy[hybrid_mask] - env.hybridization_cost
            )
            parent_b_energy[hybrid_mask] = np.maximum(
                0.0, parent_b_energy[hybrid_mask] - env.hybridization_cost
            )

    donation_a = parent_a_energy * env.sexual_offspring_energy_fraction
    donation_b = parent_b_energy * env.sexual_offspring_energy_fraction
    offspring_energy = donation_a + donation_b

    particle.energy[parent_a_idx] = parent_a_energy - donation_a
    particle.energy[parent_b_idx] = parent_b_energy - donation_b

    trait_names = list(trait_ranges.keys())
    parent_a_traits = _collect_trait_matrix(particle, parent_a_idx, trait_names)
    parent_b_traits = _collect_trait_matrix(particle, parent_b_idx, trait_names)

    offspring_matrix = _crossover_traits(
        parent_a_traits, parent_b_traits, env, trait_names
    )

    mutation_rate = env.genetics.gene_mutation_rate * env.reproduction_mutation_rate
    mutation_rate *= env.mutation_rate_sexual_multiplier * mutation_boost
    mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))

    for idx, trait in enumerate(trait_names):
        mask = np.random.random(offspring_matrix.shape[1]) < mutation_rate
        if trait == "energy_efficiency":
            mut_range = env.genetics.energy_efficiency_mutation_range
        else:
            mut_range = env.genetics.gene_mutation_range
        offspring_matrix[idx] = mutate_trait(
            offspring_matrix[idx], mask, mut_range[0], mut_range[1]
        )
        offspring_matrix[idx] = np.clip(
            offspring_matrix[idx],
            trait_ranges[trait][0],
            trait_ranges[trait][1],
        )

    offspring_traits: Dict[str, FloatArray] = {
        trait: offspring_matrix[i] for i, trait in enumerate(trait_names)
    }

    (
        offspring_traits["speed_factor"],
        offspring_traits["interaction_strength"],
        offspring_traits["perception_range"],
        offspring_traits["reproduction_rate"],
        offspring_traits["synergy_affinity"],
        offspring_traits["colony_factor"],
        offspring_traits["drift_sensitivity"],
    ) = env.genetics.clamp_gene_values(
        offspring_traits["speed_factor"],
        offspring_traits["interaction_strength"],
        offspring_traits["perception_range"],
        offspring_traits["reproduction_rate"],
        offspring_traits["synergy_affinity"],
        offspring_traits["colony_factor"],
        offspring_traits["drift_sensitivity"],
    )

    offspring_traits["energy_efficiency"] = np.clip(
        offspring_traits["energy_efficiency"],
        env.energy_efficiency_range[0],
        env.energy_efficiency_range[1],
    )

    species_ids = _assign_species_ids_for_pairs(
        particle,
        parent_a_idx,
        parent_b_idx,
        offspring_traits,
        trait_ranges,
        env,
        others,
    )

    return offspring_traits, offspring_energy, species_ids


def _produce_asexual_offspring(
    particle: "CellularTypeData",
    parent_indices: IntArray,
    env: "SimulationConfig",
    trait_ranges: Dict[str, Tuple[float, float]],
    reproduction_cost: float,
) -> Tuple[Dict[str, FloatArray], FloatArray, IntArray]:
    """Generate asexual offspring traits, energy, and species IDs."""
    parent_energy = np.maximum(
        0.0, particle.energy[parent_indices] - reproduction_cost
    )
    offspring_energy = parent_energy * env.reproduction_offspring_energy_fraction
    particle.energy[parent_indices] = parent_energy - offspring_energy

    offspring_traits = _generate_offspring_traits_vectorized(
        particle,
        parent_indices,
        env.genetics.gene_mutation_rate * env.reproduction_mutation_rate,
        env.genetics.gene_mutation_range,
        env.genetics.energy_efficiency_mutation_range,
        trait_ranges,
    )

    (
        offspring_traits["speed_factor"],
        offspring_traits["interaction_strength"],
        offspring_traits["perception_range"],
        offspring_traits["reproduction_rate"],
        offspring_traits["synergy_affinity"],
        offspring_traits["colony_factor"],
        offspring_traits["drift_sensitivity"],
    ) = env.genetics.clamp_gene_values(
        offspring_traits["speed_factor"],
        offspring_traits["interaction_strength"],
        offspring_traits["perception_range"],
        offspring_traits["reproduction_rate"],
        offspring_traits["synergy_affinity"],
        offspring_traits["colony_factor"],
        offspring_traits["drift_sensitivity"],
    )

    offspring_traits["energy_efficiency"] = np.clip(
        offspring_traits["energy_efficiency"],
        env.energy_efficiency_range[0],
        env.energy_efficiency_range[1],
    )

    species_ids = _assign_species_ids_for_asexual(
        particle, parent_indices, offspring_traits, trait_ranges, env
    )

    return offspring_traits, offspring_energy, species_ids


def _generate_offspring_traits_vectorized(
    particle: "CellularTypeData",
    parent_indices: IntArray,
    mutation_rate: float,
    mutation_range: Tuple[float, float],
    efficiency_range: Tuple[float, float],
    trait_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, FloatArray]:
    """Generate offspring traits with vectorized mutation."""
    offspring_traits: Dict[str, FloatArray] = {}
    mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))

    for trait in trait_ranges:
        parent_vals = getattr(particle, trait)[parent_indices]
        mask = np.random.random(parent_indices.size) < mutation_rate
        if trait == "energy_efficiency":
            mut_range = efficiency_range
        else:
            mut_range = mutation_range
        offspring_traits[trait] = mutate_trait(
            parent_vals, mask, mut_range[0], mut_range[1]
        )
        offspring_traits[trait] = np.clip(
            offspring_traits[trait],
            trait_ranges[trait][0],
            trait_ranges[trait][1],
        )

    return offspring_traits


def _assign_species_ids_for_pairs(
    particle: "CellularTypeData",
    parent_a_idx: IntArray,
    parent_b_idx: IntArray,
    offspring_traits: Dict[str, FloatArray],
    trait_ranges: Dict[str, Tuple[float, float]],
    env: "SimulationConfig",
    others: List["CellularTypeData"],
) -> IntArray:
    """Assign species IDs for sexual offspring with speciation checks."""
    parent_a_species = particle.species_id[parent_a_idx]
    parent_b_species = particle.species_id[parent_b_idx]
    same_species = parent_a_species == parent_b_species

    genetic_distance = _calculate_pairwise_genetic_distance(
        particle,
        parent_a_idx,
        parent_b_idx,
        offspring_traits,
        trait_ranges,
        env,
    )

    speciation_mask = genetic_distance > env.speciation_threshold
    if env.allow_hybridization:
        speciation_mask = speciation_mask | (~same_species)

    species_ids = parent_a_species.copy()
    if np.any(speciation_mask):
        next_species_id = _max_species_id(particle, others)
        for offset in np.where(speciation_mask)[0]:
            next_species_id += 1
            species_ids[offset] = next_species_id

    return species_ids


def _assign_species_ids_for_asexual(
    particle: "CellularTypeData",
    parent_indices: IntArray,
    offspring_traits: Dict[str, FloatArray],
    trait_ranges: Dict[str, Tuple[float, float]],
    env: "SimulationConfig",
) -> IntArray:
    """Assign species IDs for asexual offspring with speciation checks."""
    genetic_distance = _calculate_genetic_distance_vectorized(
        particle, parent_indices, offspring_traits, trait_ranges, env
    )
    species_ids = particle.species_id[parent_indices].copy()
    speciation_mask = genetic_distance > env.speciation_threshold

    if np.any(speciation_mask):
        next_species_id = int(np.max(particle.species_id)) if particle.species_id.size else 0
        for offset in np.where(speciation_mask)[0]:
            next_species_id += 1
            species_ids[offset] = next_species_id

    return species_ids


def _calculate_pairwise_genetic_distance(
    particle: "CellularTypeData",
    parent_a_idx: IntArray,
    parent_b_idx: IntArray,
    offspring_traits: Dict[str, FloatArray],
    trait_ranges: Dict[str, Tuple[float, float]],
    env: "SimulationConfig",
) -> FloatArray:
    """Compute genetic distance from offspring to mean of parents."""
    squared = np.zeros(parent_a_idx.size, dtype=np.float64)
    for trait in env.genetics.gene_traits:
        parent_mean = (
            getattr(particle, trait)[parent_a_idx]
            + getattr(particle, trait)[parent_b_idx]
        ) * 0.5
        offspring_vals = offspring_traits[trait]
        span = trait_ranges[trait][1] - trait_ranges[trait][0]
        if span <= 0.0:
            span = 1.0
        squared += ((offspring_vals - parent_mean) / span) ** 2
    return np.sqrt(squared)


def _calculate_genetic_distance_vectorized(
    particle: "CellularTypeData",
    parent_indices: IntArray,
    offspring_traits: Dict[str, FloatArray],
    trait_ranges: Dict[str, Tuple[float, float]],
    env: "SimulationConfig",
) -> FloatArray:
    """Calculate genetic distance for asexual reproduction in vectorized form."""
    squared = np.zeros(parent_indices.size, dtype=np.float64)
    for trait in env.genetics.gene_traits:
        parent_vals = getattr(particle, trait)[parent_indices]
        offspring_vals = offspring_traits[trait]
        span = trait_ranges[trait][1] - trait_ranges[trait][0]
        if span <= 0.0:
            span = 1.0
        squared += ((offspring_vals - parent_vals) / span) ** 2
    return np.sqrt(squared)


def _max_species_id(
    particle: "CellularTypeData", others: List["CellularTypeData"]
) -> int:
    """Compute the max species ID across all known populations."""
    max_id = int(np.max(particle.species_id)) if particle.species_id.size else particle.type_id
    for other in others:
        if other.species_id.size:
            max_id = max(max_id, int(np.max(other.species_id)))
    return max_id


def _append_offspring_bulk(
    particle: "CellularTypeData",
    parent_indices: IntArray,
    offspring_traits: Dict[str, FloatArray],
    species_ids: IntArray,
    offspring_energy: FloatArray,
    mate_indices: Optional[IntArray] = None,
) -> None:
    """Append offspring to the population using vectorized bulk operations."""
    if parent_indices.size == 0:
        return

    if mate_indices is None:
        base_x = particle.x[parent_indices]
        base_y = particle.y[parent_indices]
        base_vx = particle.vx[parent_indices]
        base_vy = particle.vy[parent_indices]
        if particle.spatial_dimensions == 3:
            base_z = particle.z[parent_indices]
            base_vz = particle.vz[parent_indices]
        else:
            base_z = np.zeros(parent_indices.size, dtype=np.float64)
            base_vz = np.zeros(parent_indices.size, dtype=np.float64)
        predation_vals = particle.predation_efficiency[parent_indices]
        mass_vals = (
            particle.mass[parent_indices].copy()
            if particle.mass_based and particle.mass is not None
            else None
        )
    else:
        base_x = (particle.x[parent_indices] + particle.x[mate_indices]) * 0.5
        base_y = (particle.y[parent_indices] + particle.y[mate_indices]) * 0.5
        base_vx = (particle.vx[parent_indices] + particle.vx[mate_indices]) * 0.5
        base_vy = (particle.vy[parent_indices] + particle.vy[mate_indices]) * 0.5
        if particle.spatial_dimensions == 3:
            base_z = (particle.z[parent_indices] + particle.z[mate_indices]) * 0.5
            base_vz = (particle.vz[parent_indices] + particle.vz[mate_indices]) * 0.5
        else:
            base_z = np.zeros(parent_indices.size, dtype=np.float64)
            base_vz = np.zeros(parent_indices.size, dtype=np.float64)
        predation_vals = (
            particle.predation_efficiency[parent_indices]
            + particle.predation_efficiency[mate_indices]
        ) * 0.5
        if particle.mass_based and particle.mass is not None:
            mass_vals = (
                particle.mass[parent_indices] + particle.mass[mate_indices]
            ) * 0.5
        else:
            mass_vals = None

    jitter_x = np.random.uniform(-5.0, 5.0, size=parent_indices.size)
    jitter_y = np.random.uniform(-5.0, 5.0, size=parent_indices.size)
    offspring_x = base_x + jitter_x
    offspring_y = base_y + jitter_y
    if particle.spatial_dimensions == 3:
        jitter_z = np.random.uniform(-5.0, 5.0, size=parent_indices.size)
        offspring_z = base_z + jitter_z
    else:
        offspring_z = np.zeros(parent_indices.size, dtype=np.float64)

    safe_efficiency = np.where(
        offspring_traits["energy_efficiency"] == 0.0,
        0.01,
        offspring_traits["energy_efficiency"],
    )
    velocity_scale = (1.0 / safe_efficiency) * offspring_traits["speed_factor"]
    offspring_vx = base_vx + np.random.uniform(-0.5, 0.5, size=parent_indices.size) * velocity_scale
    offspring_vy = base_vy + np.random.uniform(-0.5, 0.5, size=parent_indices.size) * velocity_scale
    if particle.spatial_dimensions == 3:
        offspring_vz = base_vz + np.random.uniform(-0.5, 0.5, size=parent_indices.size) * velocity_scale
    else:
        offspring_vz = np.zeros(parent_indices.size, dtype=np.float64)

    cooldown_vals = np.zeros(parent_indices.size, dtype=np.float64)
    parent_ids = np.asarray(parent_indices, dtype=np.int_)

    particle.add_components_bulk(
        x=offspring_x,
        y=offspring_y,
        vx=offspring_vx,
        vy=offspring_vy,
        energy=offspring_energy,
        mass=mass_vals,
        energy_efficiency=offspring_traits["energy_efficiency"],
        speed_factor=offspring_traits["speed_factor"],
        interaction_strength=offspring_traits["interaction_strength"],
        perception_range=offspring_traits["perception_range"],
        reproduction_rate=offspring_traits["reproduction_rate"],
        synergy_affinity=offspring_traits["synergy_affinity"],
        colony_factor=offspring_traits["colony_factor"],
        drift_sensitivity=offspring_traits["drift_sensitivity"],
        species_id=species_ids,
        parent_id=parent_ids,
        predation_efficiency=predation_vals,
        cooldown=cooldown_vals,
        z=offspring_z,
        vz=offspring_vz,
    )

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


def _get_trait_mutation_parameters(
    genetics: "GeneticParamConfig", env: "SimulationConfig"
) -> Dict[str, Tuple[float, float]]:
    """Define mutation bounds for all traits.

    Args:
        genetics: Genetics configuration containing mutation settings
        env: Simulation configuration for non-genetic trait ranges

    Returns:
        Dictionary of trait names to their (min, max) ranges
    """
    ranges: Dict[str, Tuple[float, float]] = {
        name: definition.range for name, definition in genetics.trait_definitions.items()
    }
    ranges["energy_efficiency"] = env.energy_efficiency_range
    return ranges


def _generate_offspring_traits(
    particle: "CellularTypeData",
    idx: int,
    mutation_rate: float,
    mutation_range: Tuple[float, float],
    trait_ranges: Dict[str, Tuple[float, float]],
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
    for trait_name, trait_range in trait_ranges.items():
        # Get parent trait value
        parent_value = getattr(particle, trait_name)[idx]

        # Create single-element arrays for the mutation function
        parent_array = np.array([parent_value])
        mutate_flag = np.array([mutation_rate > np.random.random()], dtype=bool)

        # Apply mutation
        offspring_traits[trait_name] = mutate_trait(
            parent_array, mutate_flag, mutation_range[0], mutation_range[1]
        )
        offspring_traits[trait_name] = np.clip(
            offspring_traits[trait_name], trait_range[0], trait_range[1]
        )

    return offspring_traits


def _calculate_genetic_distance(
    particle: "CellularTypeData",
    idx: int,
    offspring_traits: Dict[str, FloatArray],
    trait_ranges: Dict[str, Tuple[float, float]],
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

        # Normalize by trait range span
        range_min, range_max = trait_ranges.get(trait, (0.0, 1.0))
        span = range_max - range_min
        if span <= 0.0:
            span = 1.0

        # Calculate normalized squared difference
        normalized_diff = ((offspring_val - parent_val) / span) ** 2
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
    offspring_energy: float,
) -> None:
    """Add new offspring to the particle population.

    Args:
        particle: Parent particle data
        idx: Index of the parent particle
        offspring_traits: Dictionary of offspring trait values
        species_id_val: Determined species ID
        offspring_energy: Energy allocated to the offspring

    Returns:
        None: Modifies particle population in-place
    """
    # Calculate initial position with small random offset
    pos_x = float(particle.x[idx] + np.random.uniform(-5, 5))
    pos_y = float(particle.y[idx] + np.random.uniform(-5, 5))
    if particle.spatial_dimensions == 3:
        pos_z = float(particle.z[idx] + np.random.uniform(-5, 5))
    else:
        pos_z = 0.0

    # Calculate initial velocity with small random variation
    vel_x = float(particle.vx[idx] * np.random.uniform(0.9, 1.1))
    vel_y = float(particle.vy[idx] * np.random.uniform(0.9, 1.1))
    if particle.spatial_dimensions == 3:
        vel_z = float(particle.vz[idx] * np.random.uniform(0.9, 1.1))
    else:
        vel_z = 0.0

    # Use computed offspring energy
    initial_energy = float(offspring_energy)

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
        parent_id_val=int(idx),
        max_age=float(particle.max_age),
        z=pos_z,
        vz=vel_z,
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
