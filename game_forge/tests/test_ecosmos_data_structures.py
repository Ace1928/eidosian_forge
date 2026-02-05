from __future__ import annotations

import sys

from ECosmos import config as ecos_config

sys.modules.setdefault("config", ecos_config)

from ECosmos.data_structures import (
    Cell,
    Direction,
    Instruction,
    OperationType,
    Position,
    RuleSpecies,
)


def test_position_distance_and_adjacency() -> None:
    pos = Position(1, 2)
    other = Position(3, 3)
    assert pos.distance_to(other) == 3
    assert pos.adjacent_to(Position(1, 3)) is True
    assert pos.adjacent_to(Position(2, 3)) is False


def test_position_move_and_bounds() -> None:
    pos = Position(1, 1)
    moved = pos.move(Direction.NORTH)
    assert moved == Position(1, 0)
    assert moved.in_bounds(3, 3) is True
    assert Position(-1, 0).in_bounds(3, 3) is False


def test_direction_turns_and_opposite() -> None:
    assert Direction.NORTH.opposite() == Direction.SOUTH
    assert Direction.EAST.turn_left() == Direction.NORTH
    assert Direction.WEST.turn_right() == Direction.NORTH


def test_operation_costs_from_config() -> None:
    assert OperationType.MOVE.get_cost() == ecos_config.INSTRUCTION_COSTS["MOVE"]
    assert OperationType.MUTATE.get_cost() == ecos_config.INSTRUCTION_COSTS["MUTATE"]


def test_instruction_clone_and_cost() -> None:
    instr = Instruction(OperationType.SENSE, params={"radius": 3})
    clone = instr.clone()
    assert clone is not instr
    assert clone.op_type == instr.op_type
    assert clone.params == instr.params
    assert instr.get_energy_cost() == ecos_config.INSTRUCTION_COSTS["SENSE"]


def test_rule_species_basic_energy_and_rules() -> None:
    species = RuleSpecies(species_id=1, rules=[[Instruction(OperationType.MOVE)]])
    assert species.rule_usage_counts == [0]
    assert species.is_alive() is True
    species.consume_energy(species.energy + 1.0)
    assert species.is_alive() is False


def test_rule_species_add_remove_copy_rule() -> None:
    species = RuleSpecies(species_id=2, rules=[[Instruction(OperationType.MOVE)]])
    assert species.add_rule([Instruction(OperationType.SENSE)]) is True
    assert species.remove_rule(0) is True
    assert species.remove_rule(99) is False
    assert species.copy_rule(0) is True
    assert species.copy_rule(99) is False


def test_rule_species_usage_stats() -> None:
    species = RuleSpecies(species_id=3, rules=[[Instruction(OperationType.MOVE)]])
    species.record_rule_execution(0)
    species.record_rule_execution(0)
    assert species.get_most_used_rules() == [0]
    assert species.get_least_used_rules() == [0]
    species.reset_usage_stats()
    assert species.last_executed_rule == -1


def test_rule_species_energy_bounds() -> None:
    species = RuleSpecies(species_id=4, rules=[])
    species.energy = ecos_config.MAX_ENERGY - 1.0
    added = species.add_energy(10.0)
    assert species.energy == ecos_config.MAX_ENERGY
    assert added == 1.0
    assert species.get_energy_percentage() == 100.0


def test_rule_species_color_generation_and_mutation() -> None:
    base = RuleSpecies(species_id=5, rules=[])
    assert base.color == (0.4, 0.4, 0.4)
    species = RuleSpecies(species_id=6, rules=[[Instruction(OperationType.MOVE)]])
    assert all(0.0 <= channel <= 1.0 for channel in species.color)
    before = species.color
    species.mutate_color(mutation_strength=0.0)
    assert species.color == before


def test_cell_occupancy_and_resources() -> None:
    cell = Cell()
    assert cell.is_occupied() is False
    species = RuleSpecies(species_id=7, rules=[])
    assert cell.add_occupant(species) is True
    assert cell.is_occupied() is True
    assert cell.add_occupant(species) is False
    assert cell.remove_occupant() == species
    assert cell.is_occupied() is False
    cell.add_resources(5.0)
    assert cell.consume_resources(2.0) == 2.0
