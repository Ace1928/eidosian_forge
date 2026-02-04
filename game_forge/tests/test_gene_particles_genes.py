import importlib.util
import sys
import types

import numpy as np

from game_forge.src.gene_particles.gp_config import SimulationConfig
import game_forge.src.gene_particles.gp_genes as gp_genes
from game_forge.src.gene_particles.gp_genes import (
    PredationStrategy,
    _select_predation_strategy,
    _select_prey,
    _determine_species_id,
    _select_mating_pairs,
    _choose_mate_index,
    _build_linkage_indices,
    _choose_crossover_modes,
    _crossover_traits,
    _calculate_pairwise_distance,
    _calculate_pairwise_genetic_distance,
    _calculate_genetic_distance_vectorized,
    _assign_species_ids_for_pairs,
    _assign_species_ids_for_asexual,
    _get_trait_mutation_parameters,
    _max_species_id,
    _generate_offspring_traits,
    _add_offspring_to_population,
    _append_offspring_bulk,
    _calculate_attack_success,
    apply_energy_gene,
    apply_growth_gene,
    apply_interaction_gene,
    apply_movement_gene,
    apply_predation_gene,
    apply_reproduction_gene,
)
from game_forge.src.gene_particles.gp_types import CellularTypeData


def _make_particle(mass_based: bool = False) -> CellularTypeData:
    return CellularTypeData(
        type_id=0,
        color=(255, 255, 255),
        n_particles=3,
        window_width=100,
        window_height=100,
        initial_energy=200.0,
        max_age=100.0,
        mass=1.0 if mass_based else None,
    )


def _make_particle_3d(mass_based: bool = False) -> CellularTypeData:
    return CellularTypeData(
        type_id=0,
        color=(255, 255, 255),
        n_particles=3,
        window_width=100,
        window_height=100,
        initial_energy=200.0,
        max_age=100.0,
        mass=1.0 if mass_based else None,
        window_depth=120,
        spatial_dimensions=3,
    )


def test_apply_movement_interaction_energy_growth():
    env = SimulationConfig()
    ct = _make_particle()
    energy_before = ct.energy.copy()
    apply_movement_gene(ct, [1.5, 0.0, 0.0], env)
    assert np.any(ct.energy < energy_before)

    env.spatial_dimensions = 2
    apply_movement_gene(ct, [1.1, 0.0, 0.0], env)

    other = _make_particle()
    env.species_interaction_matrix = [2.0]
    env.spatial_dimensions = 3
    apply_interaction_gene(ct, [other], [0.5, 50.0], env)
    assert ct.vx.size == ct.vy.size

    env.spatial_dimensions = 2
    apply_interaction_gene(ct, [other], [0.5, 50.0], env)

    # Self-interaction and no-range branches
    apply_interaction_gene(ct, [ct], [0.5, 0.01], env)
    other_far = _make_particle()
    other_far.x[:] = 1000.0
    other_far.y[:] = 1000.0
    apply_interaction_gene(ct, [other_far], [0.5, 0.01], env)

    apply_energy_gene(ct, [0.1, 0.5, 0.3], env)
    assert np.all(ct.energy <= ct.max_energy)

    env.day_night_cycle = True
    env.time = 12.0
    env.day_length = 24.0
    env.temperature = 0.25
    apply_energy_gene(ct, [0.1, 0.5, 0.3], env)

    ct_mass = _make_particle(mass_based=True)
    mass_before = ct_mass.mass.copy()
    apply_growth_gene(ct_mass, [0.1, 1.2, 10.0], env)
    assert np.all(ct_mass.mass >= 0.0)
    assert not np.allclose(ct_mass.mass, mass_before)


def test_apply_reproduction_gene_creates_offspring():
    env = SimulationConfig()
    env.spatial_dimensions = 2
    env.sexual_reproduction_enabled = False
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 50.0, 25.0, 10.0], env)
    assert ct.x.size > initial_size


def test_reproduction_mass_based_offspring():
    env = SimulationConfig()
    env.spatial_dimensions = 2
    env.sexual_reproduction_enabled = False
    ct = _make_particle(mass_based=True)
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 50.0, 25.0, 10.0], env)
    assert ct.x.size > initial_size


def test_apply_reproduction_gene_sexual_offspring():
    np.random.seed(3)
    env = SimulationConfig()
    env.asexual_reproduction_enabled = False
    env.sexual_reproduction_probability = 1.0
    env.allow_cross_type_mating = True
    env.allow_hybridization = True
    env.hybridization_cost = 1.0
    env.compatibility_threshold = 10.0
    env.mate_selection_strategy = "compatibility"
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 250.0
    ct.x[:] = np.array([10.0, 11.0, 12.0])
    ct.y[:] = np.array([10.0, 11.0, 12.0])
    ct.species_id[:] = np.array([0, 1, 1])
    initial_energy = ct.energy.copy()
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 50.0, 25.0, 10.0, 1.0, 1.0, 1.0], env)
    assert ct.x.size > initial_size
    assert np.any(ct.energy[:initial_size] < initial_energy)


def test_append_offspring_bulk_3d_updates_z():
    env = SimulationConfig()
    env.spatial_dimensions = 3
    ct = _make_particle_3d()
    parent_indices = np.array([0], dtype=np.int_)
    offspring_traits = {
        "speed_factor": ct.speed_factor[parent_indices],
        "interaction_strength": ct.interaction_strength[parent_indices],
        "perception_range": ct.perception_range[parent_indices],
        "reproduction_rate": ct.reproduction_rate[parent_indices],
        "synergy_affinity": ct.synergy_affinity[parent_indices],
        "colony_factor": ct.colony_factor[parent_indices],
        "drift_sensitivity": ct.drift_sensitivity[parent_indices],
        "energy_efficiency": ct.energy_efficiency[parent_indices],
    }
    species_ids = np.array([0], dtype=np.int_)
    offspring_energy = np.array([10.0], dtype=np.float64)
    before = ct.z.size
    _append_offspring_bulk(ct, parent_indices, offspring_traits, species_ids, offspring_energy)
    assert ct.z.size == before + 1


def test_add_offspring_to_population_3d():
    ct = _make_particle_3d()
    offspring_traits = {
        "speed_factor": ct.speed_factor[:1],
        "interaction_strength": ct.interaction_strength[:1],
        "perception_range": ct.perception_range[:1],
        "reproduction_rate": ct.reproduction_rate[:1],
        "synergy_affinity": ct.synergy_affinity[:1],
        "colony_factor": ct.colony_factor[:1],
        "drift_sensitivity": ct.drift_sensitivity[:1],
        "energy_efficiency": ct.energy_efficiency[:1],
    }
    before = ct.z.size
    _add_offspring_to_population(ct, 0, offspring_traits, 0, 5.0)
    assert ct.z.size == before + 1


def test_predation_strategy_and_attack(monkeypatch):
    env = SimulationConfig()
    predator = _make_particle()
    prey = _make_particle()
    predator.x[:] = 0.0
    predator.y[:] = 0.0
    prey.x[:] = 1.0
    prey.y[:] = 1.0

    # Force success on first attempt
    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: 0.0)

    apply_predation_gene(predator, [prey], [1.0, 0.9, 0.0, 10.0], env)
    assert not np.all(prey.alive)

    assert _select_predation_strategy(0.0) == PredationStrategy.OPPORTUNISTIC
    assert _select_predation_strategy(1.0) == PredationStrategy.ENERGY_OPTIMAL
    assert _select_predation_strategy(2.0) == PredationStrategy.SIZE_BASED
    assert _select_predation_strategy(3.0) == PredationStrategy.TERRITORIAL

    valid_mask = np.array([True, False, True])
    distances = np.array([1.0, 5.0, 2.0])
    prey.energy[:] = np.array([1.0, 0.0, 3.0])
    selected = _select_prey(prey, valid_mask, PredationStrategy.ENERGY_OPTIMAL, distances, predator, 0)
    assert selected in (0, 2)


def test_predation_2d_distance_branch(monkeypatch):
    env = SimulationConfig()
    env.spatial_dimensions = 2
    predator = _make_particle()
    prey = _make_particle()
    predator.x[:] = 0.0
    predator.y[:] = 0.0
    prey.x[:] = 1.0
    prey.y[:] = 1.0
    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: 0.0)
    apply_predation_gene(predator, [prey], [1.0, 0.9, 0.0, 10.0], env)
    assert not np.all(prey.alive)


def test_predation_no_active_predators():
    env = SimulationConfig()
    predator = _make_particle()
    predator.energy[:] = 0.0
    prey = _make_particle()
    apply_predation_gene(predator, [prey], [1.0, 0.9, 0.0, 10.0], env)
    assert np.all(prey.alive)


def test_predation_skip_high_energy(monkeypatch):
    env = SimulationConfig()
    predator = _make_particle()
    predator.energy[:] = predator.max_energy
    prey = _make_particle()
    prey.x[:] = 0.0
    prey.y[:] = 0.0
    predator.x[:] = 0.0
    predator.y[:] = 0.0
    apply_predation_gene(predator, [prey], [1.0, 0.9, 0.0, 10.0], env)
    assert np.all(prey.alive)


def test_predation_no_valid_prey_and_none_selection():
    env = SimulationConfig()
    predator = _make_particle()
    prey = _make_particle()
    prey.energy[:] = 0.0
    # No valid prey within range
    apply_predation_gene(predator, [prey], [1.0, 0.9, 1.0, 0.01], env)
    # Valid prey but no energy so selection returns None
    prey.x[:] = predator.x
    prey.y[:] = predator.y
    apply_predation_gene(predator, [prey], [1.0, 0.9, 1.0, 10.0], env)
    assert np.all(prey.alive)


def test_predation_skips_dead_and_self():
    env = SimulationConfig()
    predator = _make_particle()
    prey = _make_particle()
    prey.alive[:] = False
    apply_predation_gene(predator, [prey, predator], [1.0, 0.9, 0.0, 10.0], env)
    assert np.all(prey.alive == False)


def test_determine_species_id_edge_cases():
    ct = _make_particle()
    ct.species_id = np.array([], dtype=int)
    species_id = _determine_species_id(ct, 0, 10.0, 0.1)
    assert species_id == 1

    ct = _make_particle()
    ct.species_id = np.array([0, 1, 2], dtype=int)
    species_id = _determine_species_id(ct, 0, 10.0, 0.1)
    assert species_id == 3


def test_genetic_distance_span_zero():
    ct = _make_particle()
    offspring_traits = {name: np.array([getattr(ct, name)[0]]) for name in [
        "speed_factor",
        "interaction_strength",
        "perception_range",
        "reproduction_rate",
        "synergy_affinity",
        "colony_factor",
        "drift_sensitivity",
    ]}
    trait_ranges = {name: (1.0, 1.0) for name in offspring_traits}
    from game_forge.src.gene_particles.gp_genes import _calculate_genetic_distance
    dist = _calculate_genetic_distance(ct, 0, offspring_traits, trait_ranges)
    assert dist >= 0.0


def test_select_prey_strategies():
    predator = _make_particle(mass_based=True)
    prey = _make_particle(mass_based=True)
    prey.mass[:] = np.array([3.0, 1.0, 2.0])
    prey.energy[:] = np.array([2.0, 1.0, 3.0])
    prey.x[:] = np.array([0.0, 5.0, 1.0])
    prey.y[:] = np.array([0.0, 5.0, 1.0])

    valid = np.array([True, True, True])
    distances = np.array([2.0, 1.0, 3.0])
    idx_size = _select_prey(prey, valid, PredationStrategy.SIZE_BASED, distances, predator, 0)
    assert idx_size in (0, 1, 2)

    idx_territory = _select_prey(prey, valid, PredationStrategy.TERRITORIAL, distances, predator, 0)
    assert idx_territory in (0, 1, 2)

    # No valid prey branch
    assert _select_prey(prey, np.array([False, False, False]), PredationStrategy.OPPORTUNISTIC, distances, predator, 0) is None

    # Energy-optimal with no energy branch
    prey.energy[:] = np.array([0.0, 0.0, 0.0])
    assert _select_prey(prey, valid, PredationStrategy.ENERGY_OPTIMAL, distances, predator, 0) is None

    # Attack success calculation with mass-based traits
    prob = _calculate_attack_success(predator, 0, prey, 0, 5.0)
    assert 0.0 <= prob <= 0.95

    # Size-based fallback to energy when mass not tracked
    prey.mass_based = False
    prey.mass = None
    idx_size_fallback = _select_prey(prey, valid, PredationStrategy.SIZE_BASED, distances, predator, 0)
    assert idx_size_fallback in (0, 1, 2)

    # Random fallback for unknown strategy
    idx_random = _select_prey(prey, valid, "unknown", distances, predator, 0)  # type: ignore[arg-type]
    assert idx_random in (0, 1, 2)


def test_choose_mate_index_strategies():
    env = SimulationConfig()
    ct = _make_particle()
    ct.energy[:] = np.array([1.0, 3.0, 2.0])
    candidates = np.array([0, 1, 2], dtype=np.int_)
    distances = np.array([0.5, 0.2, 0.9], dtype=np.float64)

    env.mate_selection_strategy = "random"
    np.random.seed(1)
    assert _choose_mate_index(ct, candidates, distances, env) in candidates

    env.mate_selection_strategy = "energy"
    assert _choose_mate_index(ct, candidates, distances, env) == 1

    env.mate_selection_strategy = "compatibility"
    assert _choose_mate_index(ct, candidates, distances, env) == 1

    env.mate_selection_strategy = "hybrid"
    assert _choose_mate_index(ct, candidates, distances, env) in candidates


def test_select_mating_pairs_and_distance_span():
    env = SimulationConfig()
    env.spatial_dimensions = 2
    env.mate_selection_strategy = "random"
    env.compatibility_threshold = 10.0
    env.allow_cross_type_mating = False
    env.mate_selection_max_neighbors = 1
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    ct.x[:] = np.array([0.0, 1.0, 2.0])
    ct.y[:] = np.array([0.0, 1.0, 2.0])
    ct.species_id[:] = np.array([0, 0, 1])

    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    for trait in env.genetics.gene_traits:
        trait_ranges[trait] = (1.0, 1.0)

    distances = _calculate_pairwise_distance(
        ct, 0, ct, np.array([1], dtype=np.int_), trait_ranges, env
    )
    assert distances.size == 1

    pairs_a, pairs_b = _select_mating_pairs(
        ct, np.array([True, True, True]), env, trait_ranges, 50.0
    )
    assert pairs_a.size == pairs_b.size


def test_choose_mate_index_empty_candidates():
    env = SimulationConfig()
    ct = _make_particle()
    env.mate_selection_strategy = "random"
    result = _choose_mate_index(
        ct, np.array([], dtype=np.int_), np.array([], dtype=np.float64), env
    )
    assert result is None


def test_crossover_modes_and_linkage_groups():
    env = SimulationConfig()
    trait_names = env.genetics.gene_traits + ["energy_efficiency"]
    parent_a = np.tile(np.arange(len(trait_names), dtype=np.float64)[:, None], (1, 3))
    parent_b = parent_a + 1.0

    linkage_groups = _build_linkage_indices(trait_names, [["speed_factor"], ["energy_efficiency"]])
    assert any(len(group) == 1 for group in linkage_groups)

    for mode in ["uniform", "arithmetic", "blend", "segment"]:
        env.crossover_mode_weights = {mode: 1.0}
        env.crossover_jitter = 0.01 if mode == "blend" else 0.0
        env.recombination_rate = 0.0 if mode == "uniform" else 0.5
        if mode == "segment":
            env.linkage_groups = [["speed_factor"], ["interaction_strength"]]
        offspring = _crossover_traits(parent_a, parent_b, env, trait_names)
        assert offspring.shape == parent_a.shape

    modes = _choose_crossover_modes({"uniform": 0.0, "blend": 0.0}, 2)
    assert len(modes) == 2

    env.crossover_mode_weights = {"segment": 1.0}
    env.linkage_groups = [trait_names]
    offspring = _crossover_traits(parent_a, parent_b, env, trait_names)
    assert offspring.shape == parent_a.shape

    env.crossover_mode_weights = {"uniform": 1.0}
    env.recombination_rate = 1.0
    env.linkage_groups = [trait_names]
    offspring = _crossover_traits(parent_a, parent_b, env, trait_names)
    assert offspring.shape == parent_a.shape


def test_species_id_assignment_helpers():
    env = SimulationConfig()
    env.allow_hybridization = True
    env.speciation_threshold = 0.01
    ct = _make_particle()
    ct.species_id[:] = np.array([0, 1, 1])
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    for trait in env.genetics.gene_traits:
        trait_ranges[trait] = (1.0, 1.0)

    parent_a = np.array([0], dtype=np.int_)
    parent_b = np.array([1], dtype=np.int_)
    offspring_traits = {
        trait: getattr(ct, trait)[parent_a] + 10.0
        for trait in env.genetics.gene_traits
    }
    offspring_traits["energy_efficiency"] = ct.energy_efficiency[parent_a]

    distances = _calculate_pairwise_genetic_distance(
        ct, parent_a, parent_b, offspring_traits, trait_ranges, env
    )
    assert distances.size == 1

    species_ids = _assign_species_ids_for_pairs(
        ct, parent_a, parent_b, offspring_traits, trait_ranges, env, []
    )
    assert species_ids.size == 1

    asexual_species_ids = _assign_species_ids_for_asexual(
        ct, parent_a, offspring_traits, trait_ranges, env
    )
    assert asexual_species_ids.size == 1

    distances_asexual = _calculate_genetic_distance_vectorized(
        ct, parent_a, offspring_traits, trait_ranges, env
    )
    assert distances_asexual.size == 1


def test_reproduction_disabled_returns():
    env = SimulationConfig()
    env.sexual_reproduction_enabled = False
    env.asexual_reproduction_enabled = False
    ct = _make_particle()
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 100.0, 25.0, 10.0], env)
    assert ct.x.size == initial_size


def test_reproduction_sexual_available_slots_zero():
    env = SimulationConfig()
    env.asexual_reproduction_enabled = False
    env.sexual_reproduction_probability = 1.0
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    env.max_particles_per_type = ct.x.size
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 100.0, 25.0, 10.0], env)
    assert ct.x.size == initial_size


def test_reproduction_sexual_truncate_pairs(monkeypatch):
    env = SimulationConfig()
    env.asexual_reproduction_enabled = False
    env.sexual_reproduction_probability = 1.0
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    env.max_particles_per_type = ct.x.size + 1

    def fake_pairs(*_args, **_kwargs):
        return np.array([0, 1], dtype=np.int_), np.array([1, 2], dtype=np.int_)

    monkeypatch.setattr(gp_genes, "_select_mating_pairs", fake_pairs)
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 50.0, 25.0, 10.0, 1.0, 1.0, 1.0], env)
    assert ct.x.size == initial_size + 1


def test_reproduction_asexual_no_candidates_after_used(monkeypatch):
    env = SimulationConfig()
    env.sexual_reproduction_probability = 1.0
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    ct.alive[2] = False

    def fake_pairs(*_args, **_kwargs):
        return np.array([0], dtype=np.int_), np.array([1], dtype=np.int_)

    monkeypatch.setattr(gp_genes, "_select_mating_pairs", fake_pairs)
    apply_reproduction_gene(ct, [], [150.0, 100.0, 25.0, 10.0, 1.0, 1.0, 1.0], env)


def test_reproduction_asexual_available_slots_zero():
    env = SimulationConfig()
    env.sexual_reproduction_enabled = False
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    env.max_particles_per_type = ct.x.size
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 100.0, 25.0, 10.0], env)
    assert ct.x.size == initial_size


def test_reproduction_asexual_truncate_parents():
    env = SimulationConfig()
    env.sexual_reproduction_enabled = False
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    env.max_particles_per_type = ct.x.size + 1
    initial_size = ct.x.size
    apply_reproduction_gene(ct, [], [150.0, 50.0, 25.0, 10.0], env)
    assert ct.x.size == initial_size + 1


def test_select_mating_pairs_single_candidate():
    env = SimulationConfig()
    ct = _make_particle()
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    pairs_a, pairs_b = _select_mating_pairs(
        ct, np.array([True, False, False]), env, trait_ranges, 50.0
    )
    assert pairs_a.size == 0
    assert pairs_b.size == 0


def test_select_mating_pairs_no_compatibility():
    env = SimulationConfig()
    env.compatibility_threshold = 0.0001
    env.allow_cross_type_mating = True
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 200.0
    ct.x[:] = np.array([0.0, 1.0, 2.0])
    ct.y[:] = np.array([0.0, 1.0, 2.0])
    ct.speed_factor[:] = np.array([0.1, 2.0, 0.1])
    ct.interaction_strength[:] = np.array([0.1, 2.0, 0.1])
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    pairs_a, pairs_b = _select_mating_pairs(
        ct, np.array([True, True, True]), env, trait_ranges, 50.0
    )
    assert pairs_a.size == 0
    assert pairs_b.size == 0


def test_select_mating_pairs_energy_none():
    env = SimulationConfig()
    env.mate_selection_strategy = "energy"
    env.compatibility_threshold = 10.0
    ct = _make_particle()
    ct.age[:] = 50.0
    ct.energy[:] = 0.0
    ct.x[:] = np.array([0.0, 1.0, 2.0])
    ct.y[:] = np.array([0.0, 1.0, 2.0])
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    pairs_a, pairs_b = _select_mating_pairs(
        ct, np.array([True, True, True]), env, trait_ranges, 50.0
    )
    assert pairs_a.size == 0
    assert pairs_b.size == 0


def test_choose_mate_index_hybrid_equal_energy():
    env = SimulationConfig()
    env.mate_selection_strategy = "hybrid"
    ct = _make_particle()
    ct.energy[:] = np.array([1.0, 1.0, 1.0])
    candidates = np.array([0, 1, 2], dtype=np.int_)
    distances = np.array([0.5, 0.2, 0.9], dtype=np.float64)
    assert _choose_mate_index(ct, candidates, distances, env) in candidates


def test_max_species_id_with_others():
    ct = _make_particle()
    other = _make_particle()
    other.species_id[:] = np.array([10, 11, 12])
    assert _max_species_id(ct, [other]) >= 10


def test_append_offspring_bulk_empty_and_mass_branch():
    env = SimulationConfig()
    ct = _make_particle(mass_based=True)
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    offspring_traits = _generate_offspring_traits(
        ct,
        0,
        env.genetics.gene_mutation_rate,
        env.genetics.gene_mutation_range,
        trait_ranges,
    )
    offspring_traits["energy_efficiency"] = np.array([ct.energy_efficiency[0]])

    _append_offspring_bulk(
        ct,
        np.array([], dtype=np.int_),
        offspring_traits,
        np.array([], dtype=np.int_),
        np.array([], dtype=np.float64),
    )

    parent_indices = np.array([0], dtype=np.int_)
    mate_indices = np.array([1], dtype=np.int_)
    _append_offspring_bulk(
        ct,
        parent_indices,
        offspring_traits,
        np.array([0], dtype=np.int_),
        np.array([1.0], dtype=np.float64),
        mate_indices=mate_indices,
    )


def test_append_offspring_bulk_mate_indices_3d():
    ct = _make_particle_3d(mass_based=True)
    env = SimulationConfig()
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    offspring_traits = _generate_offspring_traits(
        ct,
        0,
        0.0,
        (0.0, 0.0),
        trait_ranges,
    )
    offspring_traits["energy_efficiency"] = np.array([ct.energy_efficiency[0]])
    before = ct.z.size
    _append_offspring_bulk(
        ct,
        np.array([0], dtype=np.int_),
        offspring_traits,
        np.array([0], dtype=np.int_),
        np.array([1.0], dtype=np.float64),
        mate_indices=np.array([1], dtype=np.int_),
    )
    assert ct.z.size == before + 1


def test_generate_offspring_traits_and_add_offspring():
    env = SimulationConfig()
    ct = _make_particle(mass_based=True)
    trait_ranges = _get_trait_mutation_parameters(env.genetics, env)
    offspring_traits = _generate_offspring_traits(
        ct,
        0,
        env.genetics.gene_mutation_rate,
        env.genetics.gene_mutation_range,
        trait_ranges,
    )
    offspring_traits["energy_efficiency"] = np.array([ct.energy_efficiency[0]])
    species_id_val = _determine_species_id(ct, 0, 0.0, env.speciation_threshold)
    _add_offspring_to_population(ct, 0, offspring_traits, species_id_val, 1.0)


def test_kdtree_fallback_import(monkeypatch):
    module_path = "game_forge/src/gene_particles/gp_genes.py"
    dummy = types.ModuleType("scipy")
    monkeypatch.setitem(sys.modules, "scipy", dummy)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    spec = importlib.util.spec_from_file_location("gp_genes_stub", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    tree = module.KDTree(np.zeros((1, 2)))
    assert tree.query_ball_point(np.zeros((1, 2)), 1.0) == [[]]
