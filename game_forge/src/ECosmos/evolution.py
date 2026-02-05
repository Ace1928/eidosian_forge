"""
Evolution mechanisms for the ECosmos simulation.

This module handles the creation, mutation, reproduction, and interaction
of species in the evolving ecosystem.
"""

import random
import logging
from typing import List, Dict, Tuple, Optional, Set
import copy

from eidosian_core import eidosian
from data_structures import (
    RuleSpecies,
    Instruction,
    Cell,
    Position,
    Direction,
    OperationType,
)
import config

# Set up logging
logger = logging.getLogger(__name__)


@eidosian()
def create_random_instruction() -> Instruction:
    """
    Generate a random instruction with valid operation type and parameters.

    Returns:
        A randomly generated Instruction object
    """
    # Choose a random operation type
    op_type = random.choice(list(OperationType))

    # Create parameters based on operation type
    params = {}

    if op_type == OperationType.MOVE:
        params["direction"] = random.choice(list(Direction))

    elif op_type == OperationType.SENSE:
        params["sense_type"] = random.choice(["occupancy", "resources", "energy"])
        params["range"] = random.randint(1, 3)

    elif op_type == OperationType.CONSUME:
        params["target"] = random.choice(["resources", "species"])
        params["amount"] = random.uniform(1.0, 10.0)
        if params["target"] == "species":
            params["direction"] = random.choice(list(Direction))
            params["power"] = random.uniform(0.1, 0.9)

    elif op_type == OperationType.STORE:
        params["key"] = random.choice(
            ["pos", "energy", "target", "memory" + str(random.randint(0, 9))]
        )
        params["value"] = random.uniform(-10.0, 10.0)

    elif op_type == OperationType.LOAD:
        params["key"] = random.choice(
            ["pos", "energy", "target", "memory" + str(random.randint(0, 9))]
        )

    elif op_type == OperationType.CALCULATE:
        params["operation"] = random.choice(
            ["add", "subtract", "multiply", "divide", "random"]
        )
        params["operand1"] = random.uniform(-10.0, 10.0)
        params["operand2"] = random.uniform(0.1, 10.0)
        params["store_key"] = "result" + str(random.randint(0, 9))

    elif op_type == OperationType.BRANCH:
        params["condition"] = random.choice(
            ["equal", "not_equal", "greater", "less", "true", "random"]
        )
        params["value1"] = random.uniform(-10.0, 10.0)
        params["value2"] = random.uniform(-10.0, 10.0)

    elif op_type == OperationType.JUMP:
        params["rule_index"] = random.randint(0, max(0, config.INITIAL_MAX_RULES - 1))

    elif op_type == OperationType.REPRODUCE:
        # No special parameters needed
        pass

    elif op_type == OperationType.SHARE:
        params["direction"] = random.choice(list(Direction))
        params["amount"] = random.uniform(1.0, 20.0)

    elif op_type == OperationType.ATTACK:
        params["direction"] = random.choice(list(Direction))
        params["power"] = random.uniform(0.1, 1.0)

    elif op_type == OperationType.DEFEND:
        params["level"] = random.uniform(0.1, 0.9)
        params["duration"] = random.randint(1, 5)

    elif op_type == OperationType.MUTATE:
        params["rule_index"] = random.randint(0, max(0, config.INITIAL_MAX_RULES - 1))
        params["type"] = random.choice(["modify", "duplicate"])
        params["instruction_index"] = random.randint(0, 3)

    return Instruction(op_type, params)


@eidosian()
def create_random_rule() -> List[Instruction]:
    """
    Generate a random rule with 1-5 instructions.

    Returns:
        A list of Instruction objects forming a rule
    """
    instruction_count = random.randint(1, 5)
    return [create_random_instruction() for _ in range(instruction_count)]


@eidosian()
def create_random_species(species_id: int) -> RuleSpecies:
    """
    Create a new species with random properties.

    Args:
        species_id: Unique identifier for the new species

    Returns:
        A new RuleSpecies with random rules and properties
    """
    # Generate random RGB color
    color = (random.random(), random.random(), random.random())

    # Create a random set of rules (between MIN_RULES and MAX_RULES)
    rule_count = random.randint(config.INITIAL_MIN_RULES, config.INITIAL_MAX_RULES)
    rules = [create_random_rule() for _ in range(rule_count)]

    # Create the species
    species = RuleSpecies(
        species_id=species_id, rules=rules, color=color, energy=config.INITIAL_ENERGY
    )

    return species


@eidosian()
def mutate_color(species: RuleSpecies) -> None:
    """
    Slightly mutate the color of a species.

    Args:
        species: The species to mutate
    """
    if random.random() < config.COLOR_MUTATION_CHANCE:
        r, g, b = species.color

        # Small random adjustments
        dr = random.uniform(-0.1, 0.1)
        dg = random.uniform(-0.1, 0.1)
        db = random.uniform(-0.1, 0.1)

        # Ensure values stay in [0,1] range
        species.color = (
            max(0.0, min(1.0, r + dr)),
            max(0.0, min(1.0, g + dg)),
            max(0.0, min(1.0, b + db)),
        )


@eidosian()
def mutate_instruction(instruction: Instruction) -> None:
    """
    Apply mutations to an instruction.

    Args:
        instruction: The instruction to mutate
    """
    # 20% chance to change operation type completely
    if random.random() < 0.2:
        instruction.op_type = random.choice(list(OperationType))
        # Reset params since operation type changed
        instruction.params = {}

    # Mutate existing parameters
    for key, value in list(instruction.params.items()):
        if isinstance(value, (int, float)) and random.random() < 0.4:
            # Mutate numeric values
            if isinstance(value, int):
                instruction.params[key] = value + random.randint(-2, 2)
            else:  # float
                instruction.params[key] = value * (1.0 + random.uniform(-0.3, 0.3))
        elif isinstance(value, Direction) and random.random() < 0.2:
            # Change direction
            instruction.params[key] = random.choice(list(Direction))
        elif isinstance(value, str) and random.random() < 0.2:
            # For string parameters, occasionally change them
            if key == "operation" and instruction.op_type == OperationType.CALCULATE:
                instruction.params[key] = random.choice(
                    ["add", "subtract", "multiply", "divide", "random"]
                )
            elif key == "condition" and instruction.op_type == OperationType.BRANCH:
                instruction.params[key] = random.choice(
                    ["equal", "not_equal", "greater", "less", "true", "random"]
                )
            elif key.startswith("store_key") or key == "key":
                instruction.params[key] = key[0] + str(random.randint(0, 9))


@eidosian()
def mutate_species(species: RuleSpecies) -> None:
    """
    Apply mutations to a species based on mutation rate.

    Args:
        species: The species to mutate
    """
    # Get mutation rate (might be modified by species modifiers)
    mutation_rate = config.MUTATION_RATE
    if "mutation_rate_factor" in species.modifiers:
        mutation_rate *= species.modifiers["mutation_rate_factor"]

    # Check for each type of mutation
    if random.random() < mutation_rate:
        # Color mutation
        mutate_color(species)

    # Rule mutations
    for rule_idx, rule in enumerate(species.rules):
        # Chance to delete or add instructions in this rule
        if random.random() < mutation_rate and len(rule) > 1:
            # Delete an instruction
            del rule[random.randint(0, len(rule) - 1)]

        if random.random() < mutation_rate and len(rule) < config.MAX_RULE_LENGTH:
            # Add a new instruction
            rule.append(create_random_instruction())

        # Mutate existing instructions
        for instruction in rule:
            if random.random() < mutation_rate:
                mutate_instruction(instruction)

    # Add new rule
    if (
        random.random() < mutation_rate / 2
        and len(species.rules) < config.MAX_RULES_PER_SPECIES
    ):
        species.rules.append(create_random_rule())
        species.rule_usage_counts.append(0)

    # Remove rule
    if (
        random.random() < mutation_rate / 4
        and len(species.rules) > config.MIN_RULES_FOR_VIABILITY
    ):
        # Choose least used rule for deletion
        least_used_rules = species.get_least_used_rules()
        if least_used_rules:
            remove_idx = random.choice(least_used_rules)
            species.rules.pop(remove_idx)
            species.rule_usage_counts.pop(remove_idx)


@eidosian()
def reproduce_species(species: RuleSpecies, new_id: int) -> Optional[RuleSpecies]:
    """
    Create a new species from the parent species, with mutations.

    Args:
        species: The parent species
        new_id: ID for the child species

    Returns:
        A new RuleSpecies, or None if reproduction fails
    """
    if not species.can_reproduce():
        return None

    # Split energy between parent and child
    species.energy -= config.REPRODUCTION_ENERGY_COST

    # Create child species
    child = species.clone(new_id)

    # Apply mutations to child
    mutate_species(child)

    return child


@eidosian()
def handle_species_interactions(world: List[List[Cell]], x: int, y: int) -> None:
    """
    Process interactions between a species and its environment/neighbors.

    Args:
        world: 2D grid of cells
        x: X-coordinate of cell
        y: Y-coordinate of cell
    """
    cell = world[y][x]
    if cell.occupant is None:
        return

    species = cell.occupant

    # Process movement requests
    if "move_request" in species.modifiers:
        direction = species.modifiers["move_request"]
        # Clear the request
        del species.modifiers["move_request"]

        # Determine target position
        nx, ny = x, y
        if direction == 0:  # North
            ny = (y - 1) % config.WORLD_HEIGHT
        elif direction == 1:  # East
            nx = (x + 1) % config.WORLD_WIDTH
        elif direction == 2:  # South
            ny = (y + 1) % config.WORLD_HEIGHT
        elif direction == 3:  # West
            nx = (x - 1) % config.WORLD_WIDTH

        # Try to move if target cell is unoccupied
        if not world[ny][nx].is_occupied():
            # Move species to new position
            world[y][x].remove_occupant()
            world[ny][nx].add_occupant(species)
            species.position = Position(nx, ny)


@eidosian()
def inject_environmental_energy(world: List[List[Cell]], tick: int) -> None:
    """
    Inject energy into the environment to prevent extinction.

    Args:
        world: 2D grid of cells
        tick: Current time step
    """
    # Periodically add energy to low-energy species
    if tick % 10 == 0:  # Every 10 ticks
        energy_boost = config.INITIAL_ENERGY * 0.1  # 10% of initial energy

        for y in range(config.WORLD_HEIGHT):
            for x in range(config.WORLD_WIDTH):
                cell = world[y][x]
                if cell.occupant is not None:
                    species = cell.occupant

                    # Boost species with very low energy
                    if species.energy < config.MIN_VIABLE_ENERGY * 1.5:
                        species.energy += energy_boost
                        logger.debug(
                            f"Species #{species.species_id} received energy boost: +{energy_boost}"
                        )


@eidosian()
def distribute_initial_species(
    world: List[List[Cell]], next_id: int = 0, max_new: int = None
) -> int:
    """
    Randomly place initial species in the world.

    Args:
        world: 2D grid of cells
        next_id: Starting ID for new species
        max_new: Maximum number of new species to create (defaults to INITIAL_POPULATION)

    Returns:
        Number of species created
    """
    target_population = max_new if max_new is not None else config.INITIAL_POPULATION
    species_count = 0
    placement_attempts = 0
    max_attempts = target_population * 3  # Avoid infinite loop

    while species_count < target_population and placement_attempts < max_attempts:
        x = random.randint(0, config.WORLD_WIDTH - 1)
        y = random.randint(0, config.WORLD_HEIGHT - 1)

        if not world[y][x].is_occupied():
            species = create_random_species(next_id + species_count)
            species.position = Position(x, y)
            world[y][x].add_occupant(species)
            species_count += 1

        placement_attempts += 1

    return species_count  # Return number of species created
