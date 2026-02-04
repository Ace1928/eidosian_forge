import numpy as np

from game_forge.src.gene_particles.gp_config import SimulationConfig
from game_forge.src.gene_particles.gp_manager import CellularTypeManager
from game_forge.src.gene_particles.gp_types import CellularTypeData


def _make_manager(
    n_particles: int = 2,
    mass: float | None = None,
    dimensions: int = 2,
) -> tuple[CellularTypeManager, CellularTypeData]:
    config = SimulationConfig()
    config.spatial_dimensions = dimensions
    manager = CellularTypeManager(config, colors=[(1, 2, 3)], mass_based_type_indices=[])
    ct = CellularTypeData(
        type_id=0,
        color=(1, 2, 3),
        n_particles=n_particles,
        window_width=50,
        window_height=50,
        initial_energy=200.0,
        max_age=100.0,
        mass=mass,
        window_depth=60 if dimensions == 3 else None,
        spatial_dimensions=dimensions,
    )
    manager.add_cellular_type_data(ct)
    return manager, ct


def test_reproduce_skips_full_population():
    config = SimulationConfig()
    config.spatial_dimensions = 2
    config.max_particles_per_type = 1
    manager = CellularTypeManager(config, colors=[(1, 2, 3)], mass_based_type_indices=[])
    ct = CellularTypeData(
        type_id=0,
        color=(1, 2, 3),
        n_particles=1,
        window_width=50,
        window_height=50,
        initial_energy=200.0,
        max_age=100.0,
        mass=None,
        spatial_dimensions=2,
    )
    manager.add_cellular_type_data(ct)
    manager.reproduce()
    assert ct.x.size == 1


def test_reproduce_skips_mismatched_arrays():
    manager, ct = _make_manager()
    ct.y = ct.y[:1]
    manager.reproduce()
    assert ct.x.size == 2


def test_reproduce_no_offspring():
    manager, ct = _make_manager()
    ct.energy[:] = 0.0
    ct.reproduction_rate[:] = 0.0
    manager.reproduce()
    assert ct.x.size == 2


def test_reproduce_mass_mutation_branch(monkeypatch):
    manager, ct = _make_manager(mass=1.0, dimensions=2)
    manager.config.max_particles_per_type = 100
    manager.config.speciation_threshold = 0.0
    ct.energy[:] = 500.0
    ct.reproduction_rate[:] = 1.0
    ct.alive[:] = True

    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: 0.0)

    manager.reproduce()
    assert ct.x.size > 2
    assert ct.mass is not None
    assert (ct.mass > 0).all()


def test_reproduce_3d_offspring_positions():
    manager, ct = _make_manager(mass=None, dimensions=3)
    manager.config.max_particles_per_type = 100
    manager.config.speciation_threshold = 0.0
    ct.energy[:] = 500.0
    ct.reproduction_rate[:] = 1.0
    ct.alive[:] = True
    before = ct.z.size
    manager.reproduce()
    assert ct.z.size > before
    assert ct.vz.size == ct.x.size
