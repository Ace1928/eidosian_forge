from game_forge.src.gene_particles.gp_config import ReproductionMode, SimulationConfig
from game_forge.src.gene_particles.gp_interpreter import GeneticInterpreter
from game_forge.src.gene_particles.gp_types import CellularTypeData


def test_genetic_interpreter_decode():
    env = SimulationConfig()
    ct = CellularTypeData(
        type_id=0,
        color=(10, 20, 30),
        n_particles=2,
        window_width=50,
        window_height=50,
        initial_energy=50.0,
        max_age=100.0,
        mass=None,
    )
    other = CellularTypeData(
        type_id=1,
        color=(20, 30, 40),
        n_particles=2,
        window_width=50,
        window_height=50,
        initial_energy=50.0,
        max_age=100.0,
        mass=None,
    )
    interpreter = GeneticInterpreter()
    interpreter.decode(ct, [other], env)
    assert ct.vx.size == ct.vy.size


def test_genetic_interpreter_invalid_and_error_paths(monkeypatch):
    env = SimulationConfig()
    ct = CellularTypeData(
        type_id=0,
        color=(10, 20, 30),
        n_particles=1,
        window_width=10,
        window_height=10,
        initial_energy=10.0,
        max_age=10.0,
        mass=None,
    )
    interpreter = GeneticInterpreter(gene_sequence=[[], ["start_movement", 1.0]])

    def boom(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(interpreter, "_route_gene_to_handler", boom)
    interpreter.decode(ct, [], env)


def test_genetic_interpreter_reproduction_gating(monkeypatch):
    env = SimulationConfig()
    ct = CellularTypeData(
        type_id=0,
        color=(10, 20, 30),
        n_particles=1,
        window_width=10,
        window_height=10,
        initial_energy=10.0,
        max_age=10.0,
        mass=None,
    )

    called = {"count": 0}

    def mark(*_args, **_kwargs):
        called["count"] += 1

    interpreter = GeneticInterpreter(gene_sequence=[["start_reproduction", 150.0, 100.0, 50.0, 30.0]])
    monkeypatch.setattr(interpreter, "apply_reproduction_gene", mark)

    env.reproduction_mode = ReproductionMode.MANAGER
    interpreter.decode(ct, [], env)
    assert called["count"] == 0

    env.reproduction_mode = ReproductionMode.GENES
    interpreter.decode(ct, [], env)
    assert called["count"] == 1


def test_genetic_interpreter_unknown_gene():
    env = SimulationConfig()
    ct = CellularTypeData(
        type_id=0,
        color=(10, 20, 30),
        n_particles=1,
        window_width=10,
        window_height=10,
        initial_energy=10.0,
        max_age=10.0,
        mass=None,
    )
    interpreter = GeneticInterpreter(gene_sequence=[["unknown_gene", 1.0]])
    interpreter.decode(ct, [], env)
