import importlib.util
import sys
import types

import numpy as np
import pytest

from game_forge.src.gene_particles.gp_config import SimulationConfig
from game_forge.src.gene_particles.gp_types import (
    CellularTypeData,
    TraitDefinition,
    TraitType,
    random_xy,
    random_xyz,
)


def test_random_xy_bounds():
    coords = random_xy(10, 20, 5)
    assert coords.shape == (5, 2)
    assert np.all(coords[:, 0] >= 0)
    assert np.all(coords[:, 0] <= 10)
    assert np.all(coords[:, 1] >= 0)
    assert np.all(coords[:, 1] <= 20)


def test_random_xyz_bounds():
    coords = random_xyz(10, 20, 30, 4)
    assert coords.shape == (4, 3)
    assert np.all(coords[:, 0] >= 0)
    assert np.all(coords[:, 0] <= 10)
    assert np.all(coords[:, 1] >= 0)
    assert np.all(coords[:, 1] <= 20)
    assert np.all(coords[:, 2] >= 0)
    assert np.all(coords[:, 2] <= 30)


def test_cellular_type_lifecycle_and_filters():
    ct = CellularTypeData(
        type_id=0,
        color=(255, 0, 0),
        n_particles=3,
        window_width=100,
        window_height=100,
        initial_energy=10.0,
        max_age=1.0,
        mass=1.0,
    )
    ct.age = np.array([0.0, 2.0, 2.0], dtype=np.float64)
    ct.energy = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    alive_mask = ct.is_alive_mask()
    assert alive_mask.sum() == 1

    config = SimulationConfig()
    config.predation_range = 200.0
    ct.remove_dead(config)
    assert ct.x.size == 1

    mask = np.array([True], dtype=bool)
    ct.filter_by_mask(mask)
    assert ct.x.size == 1


def test_trait_definition_validation_errors():
    with pytest.raises(ValueError):
        TraitDefinition(name="", type=TraitType.MOVEMENT, range=(0.0, 1.0), description="x", default=0.5)
    with pytest.raises(ValueError):
        TraitDefinition(name="a", type=TraitType.MOVEMENT, range=(1.0,), description="x", default=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TraitDefinition(name="a", type=TraitType.MOVEMENT, range=(1.0, 0.0), description="x", default=0.5)
    with pytest.raises(ValueError):
        TraitDefinition(name="a", type=TraitType.MOVEMENT, range=(0.0, 1.0), description="", default=0.5)
    with pytest.raises(ValueError):
        TraitDefinition(name="a", type=TraitType.MOVEMENT, range=(0.0, 1.0), description="x", default=2.0)


def test_add_component_and_bulk():
    ct = CellularTypeData(
        type_id=1,
        color=(0, 255, 0),
        n_particles=1,
        window_width=50,
        window_height=50,
        initial_energy=5.0,
        max_age=10.0,
        mass=None,
    )
    initial_size = ct.x.size
    ct.add_component(
        x=10.0,
        y=10.0,
        vx=0.0,
        vy=0.0,
        z=0.0,
        vz=0.0,
        energy=5.0,
        mass_val=None,
        energy_efficiency_val=1.0,
        speed_factor_val=1.0,
        interaction_strength_val=1.0,
        perception_range_val=50.0,
        reproduction_rate_val=0.2,
        synergy_affinity_val=1.0,
        colony_factor_val=0.1,
        drift_sensitivity_val=1.0,
        species_id_val=1,
        parent_id_val=-1,
        max_age=10.0,
    )
    assert ct.x.size == initial_size + 1

    ct.add_component(
        x=12.0,
        y=12.0,
        vx=0.0,
        vy=0.0,
        z=0.0,
        vz=0.0,
        energy=5.0,
        mass_val=None,
        energy_efficiency_val=1.0,
        speed_factor_val=1.0,
        interaction_strength_val=1.0,
        perception_range_val=50.0,
        reproduction_rate_val=0.2,
        synergy_affinity_val=1.0,
        colony_factor_val=0.1,
        drift_sensitivity_val=1.0,
        species_id_val=1,
        parent_id_val=0,
        max_age=10.0,
    )
    assert ct.x.size == initial_size + 2

    ct.add_components_bulk(
        x=np.array([1.0, 2.0]),
        y=np.array([1.0, 2.0]),
        vx=np.array([0.1, 0.2]),
        vy=np.array([0.1, 0.2]),
        energy=np.array([1.0, 2.0]),
        mass=None,
        energy_efficiency=np.array([1.0, 1.0]),
        speed_factor=np.array([1.0, 1.0]),
        interaction_strength=np.array([1.0, 1.0]),
        perception_range=np.array([10.0, 10.0]),
        reproduction_rate=np.array([0.1, 0.1]),
        synergy_affinity=np.array([1.0, 1.0]),
        colony_factor=np.array([0.2, 0.2]),
        drift_sensitivity=np.array([1.0, 1.0]),
        species_id=np.array([1, 1]),
        parent_id=np.array([-1, -1]),
        z=np.array([0.0, 0.0]),
        vz=np.array([0.0, 0.0]),
    )
    assert ct.x.size == initial_size + 4
    assert ct.synergy_connections.shape[0] == ct.x.size


def test_energy_efficiency_and_mass_branches():
    ct = CellularTypeData(
        type_id=2,
        color=(10, 10, 10),
        n_particles=2,
        window_width=20,
        window_height=20,
        initial_energy=5.0,
        max_age=10.0,
        mass=1.0,
        energy_efficiency=0.5,
    )
    ct.add_component(
        x=5.0,
        y=5.0,
        vx=0.0,
        vy=0.0,
        z=0.0,
        vz=0.0,
        energy=1.0,
        mass_val=-1.0,
        energy_efficiency_val=1.0,
        speed_factor_val=1.0,
        interaction_strength_val=1.0,
        perception_range_val=10.0,
        reproduction_rate_val=0.1,
        synergy_affinity_val=1.0,
        colony_factor_val=0.1,
        drift_sensitivity_val=1.0,
        species_id_val=1,
        parent_id_val=-1,
        max_age=10.0,
    )
    assert ct.mass is not None
    assert (ct.mass > 0).all()

    with pytest.raises(ValueError):
        CellularTypeData(
            type_id=5,
            color=(0, 0, 0),
            n_particles=1,
            window_width=10,
            window_height=10,
            initial_energy=1.0,
            max_age=1.0,
            mass=0.0,
        )


def test_add_components_bulk_errors_and_empty():
    ct = CellularTypeData(
        type_id=3,
        color=(0, 0, 0),
        n_particles=1,
        window_width=10,
        window_height=10,
        initial_energy=1.0,
        max_age=5.0,
        mass=1.0,
    )
    ct.add_components_bulk(
        x=np.array([]),
        y=np.array([]),
        vx=np.array([]),
        vy=np.array([]),
        energy=np.array([]),
        mass=None,
        energy_efficiency=np.array([]),
        speed_factor=np.array([]),
        interaction_strength=np.array([]),
        perception_range=np.array([]),
        reproduction_rate=np.array([]),
        synergy_affinity=np.array([]),
        colony_factor=np.array([]),
        drift_sensitivity=np.array([]),
        species_id=np.array([], dtype=int),
        parent_id=np.array([], dtype=int),
        z=np.array([]),
        vz=np.array([]),
    )

    with pytest.raises(ValueError):
        ct.add_components_bulk(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0]),
            vx=np.array([0.0, 0.0]),
            vy=np.array([0.0, 0.0]),
            energy=np.array([1.0, 1.0]),
            mass=np.array([1.0, 1.0]),
            energy_efficiency=np.array([1.0, 1.0]),
            speed_factor=np.array([1.0, 1.0]),
            interaction_strength=np.array([1.0, 1.0]),
            perception_range=np.array([1.0, 1.0]),
            reproduction_rate=np.array([0.1, 0.1]),
            synergy_affinity=np.array([1.0, 1.0]),
            colony_factor=np.array([0.1, 0.1]),
            drift_sensitivity=np.array([1.0, 1.0]),
            species_id=np.array([1, 1]),
            parent_id=np.array([0, 0]),
            z=np.array([0.0, 0.0]),
            vz=np.array([0.0, 0.0]),
        )

    with pytest.raises(ValueError):
        ct.add_components_bulk(
            x=np.array([1.0]),
            y=np.array([1.0]),
            vx=np.array([0.0]),
            vy=np.array([0.0]),
            energy=np.array([1.0]),
            mass=np.array([1.0, 2.0]),
            energy_efficiency=np.array([1.0]),
            speed_factor=np.array([1.0]),
            interaction_strength=np.array([1.0]),
            perception_range=np.array([1.0]),
            reproduction_rate=np.array([0.1]),
            synergy_affinity=np.array([1.0]),
            colony_factor=np.array([0.1]),
            drift_sensitivity=np.array([1.0]),
            species_id=np.array([1]),
            parent_id=np.array([0]),
            z=np.array([0.0]),
            vz=np.array([0.0]),
        )


def test_internal_sync_and_energy_transfer_branches():
    ct = CellularTypeData(
        type_id=4,
        color=(1, 2, 3),
        n_particles=2,
        window_width=10,
        window_height=10,
        initial_energy=5.0,
        max_age=1.0,
        mass=None,
    )
    # Force empty array sync path by removing arrays
    for attr in [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
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
        "colony_id",
        "colony_role",
        "fitness_score",
        "generation",
        "predation_efficiency",
        "cooldown",
    ]:
        setattr(ct, attr, None)
    ct._synchronize_arrays()

    # Restore minimal arrays for energy transfer tests
    ct.x = np.array([0.0, 1.0])
    ct.y = np.array([0.0, 1.0])
    ct.z = np.array([0.0, 1.0])
    ct.energy = np.array([5.0, 5.0])
    ct.age = np.array([0.0, 0.0])
    ct.alive = np.array([True, True])
    cfg = SimulationConfig()

    # No dead by age branch
    ct._process_energy_transfer(ct.alive.copy(), cfg)

    # No alive indices branch
    ct.age = np.array([ct.max_age, ct.max_age])
    ct._process_energy_transfer(np.array([False, False]), cfg)


def test_filter_arrays_index_error_paths():
    ct = CellularTypeData(
        type_id=6,
        color=(1, 2, 3),
        n_particles=3,
        window_width=10,
        window_height=10,
        initial_energy=5.0,
        max_age=10.0,
        mass=1.0,
    )
    mask = np.array([True, False], dtype=bool)
    ct.filter_by_mask(mask)
    assert ct.x.size == 1

    ct2 = CellularTypeData(
        type_id=9,
        color=(2, 2, 2),
        n_particles=2,
        window_width=10,
        window_height=10,
        initial_energy=5.0,
        max_age=10.0,
        mass=None,
    )
    ct2.filter_by_mask(np.array([False, False], dtype=bool))
    assert ct2.mutation_history == []


def test_synchronize_trims_arrays():
    ct = CellularTypeData(
        type_id=7,
        color=(2, 3, 4),
        n_particles=2,
        window_width=10,
        window_height=10,
        initial_energy=5.0,
        max_age=10.0,
        mass=1.0,
    )
    ct.y = np.concatenate((ct.y, np.array([1.0])))
    if ct.mass is not None:
        ct.mass = np.concatenate((ct.mass, np.array([1.0])))
    ct._synchronize_arrays()
    assert ct.y.size == ct.x.size


def test_add_components_bulk_mass_branch():
    ct = CellularTypeData(
        type_id=8,
        color=(5, 5, 5),
        n_particles=1,
        window_width=10,
        window_height=10,
        initial_energy=1.0,
        max_age=10.0,
        mass=1.0,
    )
    ct.mass = None
    ct.add_components_bulk(
        x=np.array([1.0]),
        y=np.array([1.0]),
        vx=np.array([0.0]),
        vy=np.array([0.0]),
        energy=np.array([1.0]),
        mass=None,
        energy_efficiency=np.array([1.0]),
        speed_factor=np.array([1.0]),
        interaction_strength=np.array([1.0]),
        perception_range=np.array([1.0]),
        reproduction_rate=np.array([0.1]),
        synergy_affinity=np.array([1.0]),
        colony_factor=np.array([0.1]),
        drift_sensitivity=np.array([1.0]),
        species_id=np.array([1]),
        parent_id=np.array([0]),
        z=np.array([0.0]),
        vz=np.array([0.0]),
    )
    assert ct.mass is not None


def test_cellular_type_3d_initialization():
    ct = CellularTypeData(
        type_id=10,
        color=(5, 5, 5),
        n_particles=4,
        window_width=20,
        window_height=30,
        initial_energy=5.0,
        max_age=5.0,
        mass=None,
        window_depth=40,
        spatial_dimensions=3,
    )
    assert ct.z.size == ct.x.size
    assert ct.vz.size == ct.vx.size
    assert np.all(ct.z >= 0.0)
    assert np.all(ct.z <= 40.0)


def test_cellular_type_3d_default_depth_and_validation():
    ct = CellularTypeData(
        type_id=11,
        color=(1, 1, 1),
        n_particles=1,
        window_width=10,
        window_height=20,
        initial_energy=1.0,
        max_age=5.0,
        mass=None,
        spatial_dimensions=3,
    )
    assert ct.window_depth == 20

    with pytest.raises(ValueError):
        CellularTypeData(
            type_id=12,
            color=(1, 1, 1),
            n_particles=1,
            window_width=10,
            window_height=20,
            initial_energy=1.0,
            max_age=5.0,
            mass=None,
            spatial_dimensions=4,
        )

    with pytest.raises(ValueError):
        CellularTypeData(
            type_id=13,
            color=(1, 1, 1),
            n_particles=1,
            window_width=10,
            window_height=20,
            initial_energy=1.0,
            max_age=5.0,
            mass=None,
            window_depth=0,
            spatial_dimensions=3,
        )


def test_energy_transfer_3d_positions():
    ct = CellularTypeData(
        type_id=14,
        color=(2, 2, 2),
        n_particles=2,
        window_width=10,
        window_height=10,
        initial_energy=5.0,
        max_age=1.0,
        mass=None,
        window_depth=10,
        spatial_dimensions=3,
    )
    ct.age = np.array([ct.max_age, 0.0])
    ct.energy = np.array([5.0, 5.0])
    ct.alive = np.array([False, True])
    cfg = SimulationConfig()
    cfg.predation_range = 100.0
    ct._process_energy_transfer(ct.alive.copy(), cfg)


def test_add_components_bulk_defaults_for_z():
    ct = CellularTypeData(
        type_id=15,
        color=(3, 3, 3),
        n_particles=1,
        window_width=10,
        window_height=10,
        initial_energy=1.0,
        max_age=5.0,
        mass=None,
    )
    before = ct.z.size
    ct.add_components_bulk(
        x=np.array([1.0]),
        y=np.array([1.0]),
        vx=np.array([0.0]),
        vy=np.array([0.0]),
        energy=np.array([1.0]),
        mass=None,
        energy_efficiency=np.array([1.0]),
        speed_factor=np.array([1.0]),
        interaction_strength=np.array([1.0]),
        perception_range=np.array([1.0]),
        reproduction_rate=np.array([0.1]),
        synergy_affinity=np.array([1.0]),
        colony_factor=np.array([0.1]),
        drift_sensitivity=np.array([1.0]),
        species_id=np.array([1]),
        parent_id=np.array([0]),
    )
    assert ct.z.size == before + 1


def test_kdtree_stub_import(monkeypatch):
    module_path = "game_forge/src/gene_particles/gp_types.py"
    dummy = types.ModuleType("scipy")
    monkeypatch.setitem(sys.modules, "scipy", dummy)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    spec = importlib.util.spec_from_file_location("gp_types_stub", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    tree = module.KDTree(np.zeros((1, 2)))
    distances, neighbors = tree.query(np.zeros((1, 2)))
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)
