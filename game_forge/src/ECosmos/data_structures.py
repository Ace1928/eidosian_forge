"""
🧬 Core Data Structures for the ECosmos Simulation 🧬

This module defines the fundamental classes used throughout the simulation,
providing a robust foundation for building complex ecosystem behaviors.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Set, Tuple, Any, Union, TypeVar, Generic
import random
import uuid
import copy
import config
from eidosian_core import eidosian


class Position:
    """🧭 Represents a 2D position in the world grid."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y  # Fixed initialization

    @eidosian()
    def distance_to(self, other: "Position") -> int:
        """Calculate Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    @eidosian()
    def adjacent_to(self, other: "Position") -> bool:
        """Check if this position is adjacent to another position."""
        return self.distance_to(other) == 1

    @eidosian()
    def move(self, direction: "Direction") -> "Position":
        """Return a new position after moving in the specified direction."""
        offset_x, offset_y = direction.to_offset()
        return Position(self.x + offset_x, self.y + offset_y)

    @eidosian()
    def in_bounds(self, width: int, height: int) -> bool:
        """Check if this position is within the given boundaries."""
        return 0 <= self.x < width and 0 <= self.y < height

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Position({self.x}, {self.y})"


class Direction(Enum):
    """🧭 Cardinal directions for movement."""

    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    @classmethod
    def random(cls) -> "Direction":
        """Return a random direction."""
        return random.choice(list(cls))

    @eidosian()
    def to_offset(self) -> Tuple[int, int]:
        """Convert direction to x,y coordinate offset."""
        if self == Direction.NORTH:
            return (0, -1)
        elif self == Direction.EAST:
            return (1, 0)
        elif self == Direction.SOUTH:
            return (0, 1)
        elif self == Direction.WEST:
            return (-1, 0)

    @eidosian()
    def opposite(self) -> "Direction":
        """Return the opposite direction."""
        if self == Direction.NORTH:
            return Direction.SOUTH
        elif self == Direction.EAST:
            return Direction.WEST
        elif self == Direction.SOUTH:
            return Direction.NORTH
        elif self == Direction.WEST:
            return Direction.EAST

    @eidosian()
    def turn_left(self) -> "Direction":
        """Return the direction after turning left."""
        if self == Direction.NORTH:
            return Direction.WEST
        elif self == Direction.EAST:
            return Direction.NORTH
        elif self == Direction.SOUTH:
            return Direction.EAST
        elif self == Direction.WEST:
            return Direction.SOUTH

    @eidosian()
    def turn_right(self) -> "Direction":
        """Return the direction after turning right."""
        if self == Direction.NORTH:
            return Direction.EAST
        elif self == Direction.EAST:
            return Direction.SOUTH
        elif self == Direction.SOUTH:
            return Direction.WEST
        elif self == Direction.WEST:
            return Direction.NORTH


class OperationType(Enum):
    """🔧 Types of operations that species rules can perform."""

    MOVE = auto()  # Move in a direction
    SENSE = auto()  # Sense nearby environment
    CONSUME = auto()  # Consume resources or other species
    STORE = auto()  # Store data in memory
    LOAD = auto()  # Load data from memory
    CALCULATE = auto()  # Perform calculations
    BRANCH = auto()  # Conditional branching
    JUMP = auto()  # Jump to another rule
    REPRODUCE = auto()  # Trigger reproduction
    SHARE = auto()  # Share energy or information with others
    ATTACK = auto()  # Attack another species
    DEFEND = auto()  # Defend against attacks
    MUTATE = auto()  # Self-modify rules

    @eidosian()
    def get_cost(self) -> float:
        """Get the energy cost of this operation type."""
        # Convert enum name to string and look up in config
        op_name = self.name
        return config.INSTRUCTION_COSTS.get(op_name, 1.0)


class Instruction:
    """📝 A single instruction within a rule."""

    def __init__(self, op_type: OperationType, params: Dict[str, Any] = None):
        self.op_type = op_type
        self.params = params if params is not None else {}

    @eidosian()
    def clone(self) -> "Instruction":
        """Create a deep copy of this instruction."""
        return Instruction(self.op_type, params=self.params.copy())

    @eidosian()
    def get_energy_cost(self) -> float:
        """Calculate the energy cost of executing this instruction."""
        base_cost = self.op_type.get_cost()
        # Additional cost factors could be implemented here
        return base_cost

    def __repr__(self):
        return f"Instruction({self.op_type}, {self.params})"


class RuleSpecies:
    """
    🦠 Represents a species in the ecosystem with its ruleset and properties.

    A species is the fundamental unit in the ecosystem. It has:
    - Energy which it consumes to survive
    - Rules which define its behavior
    - Memory for storing and retrieving data
    - Position in the world grid
    """

    def __init__(
        self,
        species_id: int,
        rules: List[List[Instruction]] = None,
        generation: int = 0,
        energy: float = None,
        position: Position = None,
        parent_id: Optional[int] = None,
        color: Tuple[float, float, float] = None,
    ):
        self.species_id = species_id
        self.rules = rules if rules is not None else []
        self.generation = generation
        self.energy = energy if energy is not None else config.INITIAL_ENERGY
        self.position = position
        self.parent_id = parent_id

        # Memory for storing and retrieving data
        self.memory: Dict[str, Any] = {}

        # Stats for rule usage
        self.rule_usage_counts: List[int] = [0] * len(self.rules)
        self.last_executed_rule: int = -1

        # Additional properties for species
        self.modifiers: Dict[str, float] = {}

        # Color for visualization (RGB tuple with values from 0 to 1)
        self.color = color if color is not None else self._generate_color_signature()

        # Track species age in simulation ticks
        self.age: int = 0

        # Track fitness metrics
        self.resources_consumed: float = 0.0
        self.reproduction_count: int = 0
        self.attack_success_count: int = 0
        self.defense_success_count: int = 0

    def _generate_color_signature(self) -> Tuple[float, float, float]:
        """Generate a unique color based on the species rules."""
        if not self.rules:
            return (0.4, 0.4, 0.4)

        # Hash rules to get deterministic but unique colors
        hash_val = hash(str(self.rules) + str(self.species_id))
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF

        # Convert to 0-1 range for matplotlib
        return (r / 255.0, g / 255.0, b / 255.0)

    @eidosian()
    def add_rule(self, rule: List[Instruction]) -> bool:
        """Add a new rule to this species if below the maximum rule count."""
        if len(self.rules) < config.MAX_RULES_PER_SPECIES:
            self.rules.append(rule)
            self.rule_usage_counts.append(0)
            return True
        return False

    @eidosian()
    def remove_rule(self, index: int) -> bool:
        """Remove a rule at the specified index."""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
            self.rule_usage_counts.pop(index)
            return True
        return False

    @eidosian()
    def consume_energy(self, amount: float) -> bool:
        """
        Consume the specified amount of energy.
        Returns True if species remains alive, False if it dies.
        """
        self.energy -= amount
        return self.energy > 0

    @eidosian()
    def add_energy(self, amount: float) -> float:
        """
        Add energy to the species, respecting the maximum limit.
        Returns the amount of energy actually added.
        """
        before = self.energy
        self.energy = min(self.energy + amount, config.MAX_ENERGY)
        return self.energy - before

    @eidosian()
    def is_alive(self) -> bool:
        """Check if the species is still alive (has positive energy)."""
        return self.energy > 0

    @eidosian()
    def can_reproduce(self) -> bool:
        """Check if species has enough energy to reproduce."""
        return self.energy >= config.REPRODUCTION_THRESHOLD

    @eidosian()
    def reset_usage_stats(self) -> None:
        """Reset rule usage statistics for a new cycle."""
        self.rule_usage_counts = [0] * len(self.rules)
        self.last_executed_rule = -1

    @eidosian()
    def record_rule_execution(self, rule_index: int) -> None:
        """Record that a rule was executed."""
        if 0 <= rule_index < len(self.rule_usage_counts):
            self.rule_usage_counts[rule_index] += 1
            self.last_executed_rule = rule_index

    @eidosian()
    def get_least_used_rules(self) -> List[int]:
        """Get indices of the least frequently used rules."""
        if not self.rule_usage_counts:
            return []

        min_usage = min(self.rule_usage_counts)
        return [
            i for i, usage in enumerate(self.rule_usage_counts) if usage == min_usage
        ]

    @eidosian()
    def get_most_used_rules(self) -> List[int]:
        """Get indices of the most frequently used rules."""
        if not self.rule_usage_counts:
            return []

        max_usage = max(self.rule_usage_counts)
        return [
            i for i, usage in enumerate(self.rule_usage_counts) if usage == max_usage
        ]

    @eidosian()
    def copy_rule(self, source_index: int) -> bool:
        """Copy an existing rule if below the maximum rule count."""
        if source_index < 0 or source_index >= len(self.rules):
            return False

        if len(self.rules) < config.MAX_RULES_PER_SPECIES:
            # Deep copy the rule
            new_rule = [instruction.clone() for instruction in self.rules[source_index]]
            self.rules.append(new_rule)
            self.rule_usage_counts.append(0)
            return True
        return False

    @eidosian()
    def get_energy_percentage(self) -> float:
        """Get energy as a percentage of maximum energy."""
        return min(1.0, self.energy / config.MAX_ENERGY) * 100

    @eidosian()
    def mutate_color(self, mutation_strength: float = 0.1) -> None:
        """Slightly modify the species color."""
        r, g, b = self.color
        r = max(0, min(1, r + (random.random() - 0.5) * mutation_strength))
        g = max(0, min(1, g + (random.random() - 0.5) * mutation_strength))
        b = max(0, min(1, b + (random.random() - 0.5) * mutation_strength))
        self.color = (r, g, b)

    @eidosian()
    def age_tick(self) -> None:
        """Increment the age counter of this species."""
        self.age += 1

    @eidosian()
    def get_fitness_score(self) -> float:
        """Calculate a fitness score based on various metrics."""
        return (
            self.energy * 0.5
            + self.resources_consumed * 0.2
            + self.reproduction_count * 5.0
            + self.attack_success_count * 1.0
            + self.defense_success_count * 0.5
        )

    @eidosian()
    def clone(self, new_id: int) -> "RuleSpecies":
        """Create a clone of this species with a new ID."""
        # Deep copy all rules
        cloned_rules = []
        for rule in self.rules:
            cloned_rule = [instruction.clone() for instruction in rule]
            cloned_rules.append(cloned_rule)

        # Create new species with incremented generation
        return RuleSpecies(
            species_id=new_id,
            rules=cloned_rules,
            generation=self.generation + 1,
            energy=config.INITIAL_ENERGY,
            parent_id=self.species_id,
            color=self.color,  # Inherit parent's color (mutations handled separately)
        )

    def __repr__(self):
        return f"Species(id={self.species_id}, gen={self.generation}, rules={len(self.rules)}, energy={self.energy:.1f})"


class Cell:
    """🧩 A single cell in the world grid."""

    def __init__(self, occupant: Optional[RuleSpecies] = None):
        self.occupant = occupant
        self.resources: float = 0.0
        self.last_modified_tick: int = 0
        self.terrain_type: str = "normal"  # Could be: normal, water, mountain, etc.
        self.environmental_factors: Dict[str, float] = {}  # Store environmental values

    @eidosian()
    def is_occupied(self) -> bool:
        """Check if cell is currently occupied."""
        return self.occupant is not None

    @eidosian()
    def add_occupant(self, species: RuleSpecies) -> bool:
        """Add a species to this cell if it's empty."""
        if not self.is_occupied():
            self.occupant = species
            return True
        return False

    @eidosian()
    def remove_occupant(self) -> Optional[RuleSpecies]:
        """Remove and return the current occupant."""
        occupant = self.occupant
        self.occupant = None
        return occupant

    @eidosian()
    def add_resources(self, amount: float) -> None:
        """Add resources to this cell."""
        self.resources += amount

    @eidosian()
    def consume_resources(self, amount: float) -> float:
        """
        Consume resources from this cell.
        Returns the amount actually consumed.
        """
        consumed = min(amount, self.resources)
        self.resources -= consumed
        return consumed

    @eidosian()
    def update(self, current_tick: int, resource_regen_rate: float) -> None:
        """
        Update cell state based on time.
        Handles resource regeneration and other time-based effects.
        """
        ticks_elapsed = current_tick - self.last_modified_tick

        # Regenerate resources based on time elapsed
        if ticks_elapsed > 0:
            self.resources += resource_regen_rate * ticks_elapsed
            self.last_modified_tick = current_tick

    @eidosian()
    def set_environmental_factor(self, factor_name: str, value: float) -> None:
        """Set an environmental factor value for this cell."""
        self.environmental_factors[factor_name] = value

    @eidosian()
    def get_environmental_factor(self, factor_name: str) -> float:
        """Get the value of an environmental factor, defaulting to 0."""
        return self.environmental_factors.get(factor_name, 0.0)

    @eidosian()
    def set_terrain_type(self, terrain_type: str) -> None:
        """Set the terrain type of this cell."""
        self.terrain_type = terrain_type

    def __repr__(self):
        if self.occupant:
            return f"Cell(species={self.occupant.species_id}, resources={self.resources:.1f})"
        else:
            return f"Cell(empty, resources={self.resources:.1f})"
