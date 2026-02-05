import pytest

from game_forge.src.gene_particles.gp_config import (
    GeneticParamConfig,
    ReproductionMode,
    SimulationConfig,
)


def test_genetic_param_config_defaults():
    config = GeneticParamConfig()
    for trait in config.CORE_TRAITS:
        assert trait in config.trait_definitions
    assert len(config.gene_traits) == len(config.trait_definitions)


def test_genetic_param_config_invalid_probability():
    config = GeneticParamConfig()
    config.gene_mutation_rate = 1.5
    with pytest.raises(ValueError):
        config._validate()


def test_genetic_param_config_invalid_range():
    config = GeneticParamConfig()
    config.gene_mutation_range = (0.1, 0.0)
    with pytest.raises(ValueError):
        config._validate()


def test_genetic_param_config_round_trip():
    config = GeneticParamConfig()
    payload = config.to_dict()
    restored = GeneticParamConfig.from_dict(payload)
    assert restored.trait_definitions.keys() == config.trait_definitions.keys()


def test_genetic_param_config_get_range():
    config = GeneticParamConfig()
    speed_range = config.get_range_for_trait("speed_factor")
    assert speed_range[0] < speed_range[1]
    with pytest.raises(KeyError):
        config.get_range_for_trait("unknown_trait")


def test_genetic_param_config_missing_trait_definition():
    config = GeneticParamConfig()
    config.trait_definitions = {}
    config.gene_traits = ["speed_factor"]
    with pytest.raises(ValueError):
        config._validate()


def test_genetic_param_config_from_dict_string_trait_type():
    config = GeneticParamConfig()
    payload = config.to_dict()
    trait_def = payload["trait_definitions"]["speed_factor"]
    trait_def["type"] = "MOVEMENT"
    restored = GeneticParamConfig.from_dict(payload)
    assert restored.trait_definitions["speed_factor"].type.name == "MOVEMENT"


def test_simulation_config_round_trip():
    config = SimulationConfig()
    payload = config.to_dict()
    restored = SimulationConfig.from_dict(payload)
    assert restored.n_cell_types == config.n_cell_types
    assert restored.genetics.gene_traits == config.genetics.gene_traits
    assert restored.reproduction_mode == ReproductionMode.MANAGER
    assert restored.spatial_dimensions == config.spatial_dimensions
    assert restored.projection_mode == config.projection_mode
    assert restored.boundary_mode == config.boundary_mode


def test_simulation_config_environment_advance():
    config = SimulationConfig()
    config.day_night_cycle = True
    config.time = 0.0
    config.time_step = 2.0
    config.temperature = 0.5
    config.temperature_drift = 0.1
    config.temperature_noise = 0.05
    config.temperature_bounds = (0.0, 1.0)
    called = {"count": 0}

    def hook(cfg, frame):
        _ = cfg, frame
        called["count"] += 1

    config.environment_hooks = [hook]
    config.advance_environment(1)
    assert config.time == 2.0
    assert config.temperature >= 0.5
    assert called["count"] == 1


def test_simulation_config_reproduction_mode_validation():
    config = SimulationConfig()
    config.reproduction_mode = "manager"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        config._validate()


def test_simulation_config_invalid_crossover_mode():
    config = SimulationConfig()
    config.crossover_mode_weights = {"invalid": 1.0}
    with pytest.raises(ValueError):
        config._validate()


def test_simulation_config_invalid_linkage_groups():
    config = SimulationConfig()
    config.linkage_groups = []
    with pytest.raises(ValueError):
        config._validate()

    config = SimulationConfig()
    config.linkage_groups = [[]]
    with pytest.raises(ValueError):
        config._validate()

    config = SimulationConfig()
    config.linkage_groups = [["unknown_trait"]]
    with pytest.raises(ValueError):
        config._validate()


def test_simulation_config_from_dict_reproduction_mode_branches():
    config = SimulationConfig()
    payload = config.to_dict()
    payload["reproduction_mode"] = ReproductionMode.GENES
    restored = SimulationConfig.from_dict(payload)
    assert restored.reproduction_mode == ReproductionMode.GENES

    payload["reproduction_mode"] = 123
    with pytest.raises(ValueError):
        SimulationConfig.from_dict(payload)


def test_simulation_config_validation_errors():
    config = SimulationConfig()
    config.n_cell_types = 0
    with pytest.raises(ValueError):
        config._validate()

    config = SimulationConfig()
    config.mass_range = (1.0, 1.0)
    with pytest.raises(ValueError):
        config._validate()

    config = SimulationConfig()
    config.friction = 2.0
    with pytest.raises(ValueError):
        config._validate()

    config = SimulationConfig()
    config.synergy_range = -1.0
    with pytest.raises(ValueError):
        config._validate()


def test_simulation_config_all_validation_branches():
    invalid_cases = [
        ("particles_per_type", 0),
        ("min_particles_per_type", 0),
        ("max_particles_per_type", 0),
        ("mass_range", (0.0, 1.0)),
        ("mass_range", (1.0, 0.5)),
        ("base_velocity_scale", 0.0),
        ("mass_based_fraction", -0.1),
        ("interaction_strength_range", (1.0, 0.5)),
        ("initial_energy", 0.0),
        ("friction", -0.1),
        ("global_temperature", -0.1),
        ("predation_range", 0.0),
        ("energy_transfer_factor", 1.5),
        ("energy_efficiency_range", (1.0, 0.5)),
        ("max_energy", 0.0),
        ("spatial_dimensions", 1),
        ("world_depth", 0.0),
        ("world_width", 0.0),
        ("world_height", 0.0),
        ("boundary_mode", "bounce"),
        ("projection_mode", "tilt"),
        ("projection_distance", 0.0),
        ("depth_fade_strength", -0.1),
        ("depth_min_scale", 0.0),
        ("depth_max_scale", 0.1),
        ("max_frames", -1),
        ("evolution_interval", 0),
        ("day_length", 0.0),
        ("time_step", 0.0),
        ("temperature_bounds", (1.0, 0.0)),
        ("temperature", -0.1),
        ("temperature_noise", -0.1),
        ("gene_interpreter_interval", 0),
        ("use_force_registry", "yes"),
        ("force_registry_min_particles", 0),
        ("reproduction_energy_threshold", 0.0),
        ("reproduction_mutation_rate", -0.1),
        ("reproduction_offspring_energy_fraction", 1.5),
        ("sexual_reproduction_probability", 1.5),
        ("sexual_offspring_energy_fraction", 1.5),
        ("max_offspring_per_pair", 0),
        ("mate_selection_radius", 0.0),
        ("mate_selection_max_neighbors", 0),
        ("compatibility_threshold", 0.0),
        ("compatibility_weight", -0.1),
        ("crossover_mode_weights", {}),
        ("crossover_blend_alpha", -0.1),
        ("recombination_rate", 1.5),
        ("hybridization_cost", -0.1),
        ("mutation_rate_sexual_multiplier", 0.0),
        ("mate_selection_strategy", "unknown"),
        ("cluster_radius", 0.0),
        ("particle_size", 0.0),
        ("speciation_threshold", 0.0),
        ("colony_formation_probability", 1.5),
        ("colony_radius", 0.0),
        ("colony_cohesion_strength", -0.1),
        ("synergy_evolution_rate", 1.5),
        ("complexity_factor", 0.0),
        ("structural_complexity_weight", -0.1),
    ]

    for attr, value in invalid_cases:
        cfg = SimulationConfig()
        setattr(cfg, attr, value)
        with pytest.raises(ValueError):
            cfg._validate()
