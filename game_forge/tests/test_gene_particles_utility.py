import numpy as np

from game_forge.src.gene_particles import gp_utility
from game_forge.src.gene_particles.gp_config import SimulationConfig


def test_mutate_trait_scalar_and_vector():
    assert gp_utility.mutate_trait(1.0, False, -0.1, 0.1) == 1.0

    np.random.seed(0)
    mutated_scalar = gp_utility.mutate_trait(1.0, True, -0.5, 0.5)
    assert mutated_scalar != 1.0

    base = np.array([1.0, 1.0, 1.0])
    mutated = gp_utility.mutate_trait(base, np.array([True, False, True]), -0.1, 0.1)
    assert mutated.shape == base.shape
    no_change = gp_utility.mutate_trait(base, True, 0.0, 0.0)
    assert np.allclose(no_change, base)
    no_mut = gp_utility.mutate_trait(base, False, -0.1, 0.1)
    assert np.allclose(no_mut, base)


def test_random_xy_and_growth():
    coords = gp_utility.random_xy(10, 10, 2)
    assert coords.shape == (2, 2)

    energy = np.array([1.0, 2.0])
    grown = gp_utility.apply_growth_gene(energy, 2.0, 0.5, 10.0)
    assert np.all(grown >= 0.5)


def test_generate_colors_and_interactions():
    colors = gp_utility.generate_vibrant_colors(6)
    assert len(colors) == 6
    for r, g, b in colors:
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    params = {
        "max_dist": 10.0,
        "use_potential": True,
        "potential_strength": 1.0,
        "use_gravity": True,
        "m_a": 1.0,
        "m_b": 1.0,
        "gravity_factor": 1.0,
    }
    fx, fy = gp_utility.apply_interaction(0.0, 0.0, 1.0, 1.0, params)
    assert isinstance(fx, float)
    assert isinstance(fy, float)
    fx, fy = gp_utility.apply_interaction(0.0, 0.0, 100.0, 100.0, {"max_dist": 1.0})
    assert fx == 0.0 and fy == 0.0


def test_energy_transfer_and_synergy():
    config = SimulationConfig()
    giver = np.array([10.0, 20.0])
    receiver = np.array([5.0, 5.0])
    giver_mass = np.array([1.0, 1.0])
    receiver_mass = np.array([1.0, 1.0])

    new_giver, new_receiver, new_giver_mass, new_receiver_mass = gp_utility.give_take_interaction(
        giver, receiver, giver_mass, receiver_mass, config
    )
    assert np.all(new_giver >= giver)
    assert np.all(new_receiver <= receiver)
    assert new_giver_mass is not None
    assert new_receiver_mass is not None

    a, b = gp_utility.apply_synergy(np.array([1.0]), np.array([3.0]), 1.0)
    assert float(a[0]) == float(b[0])


def test_mutate_trait_invalid_type():
    try:
        gp_utility.mutate_trait("bad", True, -0.1, 0.1)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "Mutation requires" in str(exc)


def test_module_demo_executes(capsys):
    gp_utility.demo()
    captured = capsys.readouterr()
    assert "Gene Particles Utility Module" in captured.out
