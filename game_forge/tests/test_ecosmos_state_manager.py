from __future__ import annotations

import sys
from pathlib import Path

from ECosmos import config as ecos_config

sys.modules.setdefault("config", ecos_config)

from ECosmos import data_structures as ecos_data_structures

sys.modules.setdefault("data_structures", ecos_data_structures)

from ECosmos.data_structures import Cell, RuleSpecies, Position
from ECosmos.state_manager import StateManager


def test_state_manager_save_and_load(tmp_path: Path) -> None:
    manager = StateManager(state_dir=str(tmp_path))
    world = [[Cell(RuleSpecies(1, position=Position(0, 0)))]]
    stats = {"Population": 1}
    saved = manager.save_state(world, tick=5, next_species_id=2, stats=stats, filename="state.pkl")
    assert Path(saved).exists()
    loaded = manager.load_state("state.pkl")
    assert loaded is not None
    assert loaded["tick"] == 5
    assert loaded["next_species_id"] == 2
    assert loaded["stats"]["Population"] == 1


def test_state_manager_list_states(tmp_path: Path) -> None:
    manager = StateManager(state_dir=str(tmp_path))
    manager.save_state([[Cell()]], tick=1, next_species_id=1, stats={}, filename="a.pkl")
    manager.save_state([[Cell()]], tick=2, next_species_id=2, stats={}, filename="b.pkl")
    states = manager.list_available_states()
    filenames = [name for name, _ in states]
    assert "a.pkl" in filenames
    assert "b.pkl" in filenames
