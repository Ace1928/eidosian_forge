import random

import numpy as np

from game_forge.src.gene_particles.gp_config import SimulationConfig, MIN_INTERACTION_DISTANCE
from game_forge.src.gene_particles.gp_rules import InteractionRules


def test_interaction_rules_creation_and_evolution(monkeypatch):
    config = SimulationConfig()
    config.n_cell_types = 2
    config.mass_based_fraction = 1.0
    rules = InteractionRules(config, [0, 1])
    assert len(rules.rules) == config.n_cell_types ** 2
    assert rules.give_take_matrix.shape == (2, 2)
    assert rules.synergy_matrix.shape == (2, 2)

    # Force evolution to trigger
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "uniform", lambda a, b: (a + b) / 2)

    rules.evolve_parameters(config.evolution_interval)
    assert 0.0 <= config.energy_transfer_factor <= 1.0
    assert np.all(rules.synergy_matrix >= 0.0)
    assert np.all(rules.synergy_matrix <= 1.0)

    for _, _, params in rules.rules:
        assert params["max_dist"] >= MIN_INTERACTION_DISTANCE


def test_evolve_gravity_factor_branch(monkeypatch):
    config = SimulationConfig()
    config.n_cell_types = 1
    config.mass_based_fraction = 1.0

    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "uniform", lambda a, b: a)

    rules = InteractionRules(config, [0])
    gravity_before = [params["gravity_factor"] for _, _, params in rules.rules]
    rules.evolve_parameters(config.evolution_interval)
    gravity_after = [params["gravity_factor"] for _, _, params in rules.rules]
    assert gravity_after != gravity_before
