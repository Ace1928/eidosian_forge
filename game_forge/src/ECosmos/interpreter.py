"""
Interpreter for species rules in the ECosmos simulation.

This module handles the execution of rules that define species behavior.
"""

import random
import time
import logging
from typing import List, Dict, Tuple, Any, Optional, Set

from data_structures import RuleSpecies, Instruction, OperationType, Direction, Position
import config

# Set up logging
logger = logging.getLogger(__name__)

# Global reference to the world grid for interaction operations
# This will be set by the main simulation before running species
world_grid = None
world_width = None
world_height = None


def set_world_reference(grid, width, height):
    """Set global reference to the world grid for the interpreter."""
    global world_grid, world_width, world_height
    world_grid = grid
    world_width = width
    world_height = height


def get_neighbor_position(position: Position, direction: Direction) -> Position:
    """
    Get neighboring position in the given direction with wraparound.

    Args:
        position: Current position
        direction: Direction to move

    Returns:
        New position with wraparound applied
    """
    if position is None or direction is None:
        # Return a default position if inputs are invalid
        return Position(0, 0)

    offset_x, offset_y = direction.to_offset()
    new_x = safe_modulo(position.x + offset_x, world_width)
    new_y = safe_modulo(position.y + offset_y, world_height)
    return Position(new_x, new_y)


def is_valid_position(x: int, y: int) -> bool:
    """Check if position is within world boundaries."""
    if x is None or y is None:
        return False
    return 0 <= x < world_width and 0 <= y < world_height


def safe_int_cast(value):
    """Safely convert a value to integer."""
    try:
        if value is None:
            return 0
        return int(value)
    except (ValueError, TypeError):
        return 0


def safe_modulo(a, b):
    """Safely perform modulo operation."""
    try:
        if b is None or b == 0:
            return 0
        a = 0 if a is None else a
        return a % b
    except (TypeError, ValueError):
        return 0


def safe_index(lst, idx):
    """Safely access a list element by index."""
    try:
        if lst is None:
            return None

        # Convert float indices to integers
        if isinstance(idx, float):
            idx = safe_int_cast(idx)

        # Check bounds
        if 0 <= idx < len(lst):
            return lst[idx]
        return None
    except (TypeError, IndexError):
        return None


def safe_execute_instruction(instruction_type, func, *args, **kwargs):
    """
    Safely execute an instruction with error handling.

    Args:
        instruction_type: The type of operation being performed
        func: The function to execute
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Result of the function or None if execution failed
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing instruction {instruction_type}: {str(e)}")
        return None


def handle_move_operation(species, *args):
    # Example implementation with safe handling
    direction = safe_int_cast(args[0]) if args else 0
    distance = safe_int_cast(args[1]) if len(args) > 1 else 1

    # Calculate new position safely
    current_x = species.position.x if hasattr(species, "position") else 0
    current_y = species.position.y if hasattr(species, "position") else 0

    # Use safe modulo for world boundaries
    new_x = safe_modulo(current_x + (direction % 2) * distance, species.world_width)
    new_y = safe_modulo(
        current_y + ((direction + 1) % 2) * distance, species.world_height
    )

    # Update position
    if hasattr(species, "position"):
        species.position.x = new_x
        species.position.y = new_y

    return True


def handle_consume_operation(species, *args):
    # Example implementation with safe handling
    target = args[0] if args else None

    # Safe access to resources or targets
    if target is not None:
        # Use safe_index for any list accessing
        resource = safe_index(species.resources, target)
        # Further processing...

    return True


def handle_calculate_operation(species, *args):
    # Example implementation with safe handling
    if len(args) < 3:
        return False

    operand1 = args[0] if args[0] is not None else 0
    operator = args[1] if args[1] is not None else 0
    operand2 = args[2] if args[2] is not None else 0

    # Ensure integers for operations requiring them
    if operator == 3:  # Assuming 3 is integer division
        operand1 = safe_int_cast(operand1)
        operand2 = safe_int_cast(operand2)

    # Perform calculation with safety checks
    # ...

    return True


def execute_instruction(
    species: RuleSpecies, instruction: Instruction
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Execute a single instruction for a species.

    Args:
        species: The species executing the instruction
        instruction: The instruction to execute

    Returns:
        Tuple of (success, result_data)
    """
    op_type = instruction.op_type
    params = instruction.params if instruction.params else {}

    # Default energy consumption
    energy_cost = config.RULE_EXECUTION_COST
    result_data = None
    success = False

    try:
        # Handle different operation types
        if op_type == OperationType.MOVE:
            # Move in a direction
            direction = params.get("direction")
            if direction is None:
                # Random direction if not specified
                direction = Direction.random()

            # Calculate new position with wraparound
            if not hasattr(species, "position") or species.position is None:
                species.position = Position(0, 0)

            new_pos = get_neighbor_position(species.position, direction)

            # Check if target cell is empty and world_grid is properly initialized
            if (
                world_grid is not None
                and 0 <= new_pos.y < len(world_grid)
                and 0 <= new_pos.x < len(world_grid[0])
                and not world_grid[new_pos.y][new_pos.x].is_occupied()
            ):

                # Remove from current cell and add to new cell
                current_x, current_y = species.position.x, species.position.y
                if 0 <= current_y < len(world_grid) and 0 <= current_x < len(
                    world_grid[0]
                ):
                    world_grid[current_y][current_x].remove_occupant()

                world_grid[new_pos.y][new_pos.x].add_occupant(species)
                species.position = new_pos
                success = True
                energy_cost *= 2  # Moving costs more energy

                # Return new position
                result_data = {"new_position": new_pos}
            else:
                # Target cell is occupied or invalid
                result_data = {"occupied": True}

        elif op_type == OperationType.SENSE:
            # Sense environment or nearby cells
            sense_type = params.get("sense_type", "occupancy")
            range_val = safe_int_cast(params.get("range", 1))

            # Limit sensing range
            range_val = min(range_val, 3)

            results = {}
            if not hasattr(species, "position") or species.position is None:
                species.position = Position(0, 0)

            x, y = species.position.x, species.position.y

            # Check if world_grid is properly initialized
            if world_grid is None:
                result_data = {"error": "world not initialized"}
                success = False
                return success, result_data

            # Scan cells within range
            for dy in range(-range_val, range_val + 1):
                for dx in range(-range_val, range_val + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip self

                    scan_x = safe_modulo(x + dx, world_width)
                    scan_y = safe_modulo(y + dy, world_height)

                    # Validate indices before accessing world_grid
                    if 0 <= scan_y < len(world_grid) and 0 <= scan_x < len(
                        world_grid[0]
                    ):
                        cell = world_grid[scan_y][scan_x]

                        if sense_type == "occupancy":
                            results[(dx, dy)] = cell.is_occupied()
                        elif sense_type == "resources":
                            results[(dx, dy)] = cell.resources
                        elif sense_type == "energy" and cell.occupant:
                            results[(dx, dy)] = cell.occupant.energy

            success = True
            result_data = {"sense_results": results}

        elif op_type == OperationType.CONSUME:
            # Consume resources or attack other species
            target_type = params.get("target", "resources")
            amount = (
                float(params.get("amount", 5.0))
                if params.get("amount") is not None
                else 5.0
            )

            if target_type == "resources":
                # Consume resources from current cell
                if (
                    hasattr(species, "position")
                    and species.position is not None
                    and world_grid is not None
                ):
                    pos_x, pos_y = species.position.x, species.position.y

                    if 0 <= pos_y < len(world_grid) and 0 <= pos_x < len(world_grid[0]):
                        cell = world_grid[pos_y][pos_x]
                        consumed = cell.consume_resources(amount)
                        species.energy += consumed
                        success = consumed > 0
                        result_data = {"consumed": consumed}
                    else:
                        result_data = {"error": "invalid position"}
                else:
                    result_data = {"error": "no position or world"}

            elif target_type == "species":
                # Try to consume energy from a neighboring species (attack)
                if not hasattr(species, "position") or species.position is None:
                    species.position = Position(0, 0)

                direction = params.get("direction", Direction.random())
                target_pos = get_neighbor_position(species.position, direction)

                # Validate indices before accessing world_grid
                if (
                    world_grid is not None
                    and 0 <= target_pos.y < len(world_grid)
                    and 0 <= target_pos.x < len(world_grid[0])
                ):

                    target_cell = world_grid[target_pos.y][target_pos.x]

                    if target_cell.occupant:
                        # Attack success chance based on relative energy
                        attack_power = (
                            float(params.get("power", 0.5))
                            if params.get("power") is not None
                            else 0.5
                        )
                        success_chance = min(
                            0.8,
                            species.energy
                            / (target_cell.occupant.energy + species.energy),
                        )

                        if random.random() < success_chance:
                            # Attack succeeds
                            steal_amount = min(
                                amount, target_cell.occupant.energy * attack_power
                            )
                            target_cell.occupant.energy -= steal_amount
                            species.energy += (
                                steal_amount * 0.8
                            )  # 80% efficiency in energy transfer
                            success = True
                            result_data = {"stolen": steal_amount}

                            # Check if target died from attack
                            if target_cell.occupant.energy <= 0:
                                target_cell.remove_occupant()
                        else:
                            # Attack failed
                            result_data = {"failed": True}
                            energy_cost *= 2  # Failed attacks are costly
                    else:
                        result_data = {"no_target": True}
                else:
                    result_data = {"error": "invalid position or world"}

        elif op_type == OperationType.STORE:
            # Store data in species memory
            key = str(params.get("key", "default"))
            value = params.get("value")

            species.memory[key] = value
            success = True
            result_data = {"stored": key}

        elif op_type == OperationType.LOAD:
            # Load data from species memory
            key = str(params.get("key", "default"))

            if key in species.memory:
                result_data = {"value": species.memory[key]}
                success = True
            else:
                result_data = {"not_found": True}

        elif op_type == OperationType.CALCULATE:
            # Perform calculations
            operation = str(params.get("operation", "add"))
            operand1 = params.get("operand1", 0)
            operand2 = params.get("operand2", 0)
            store_key = params.get("store_key")

            # Ensure operands have appropriate types
            try:
                if operation in ["random", "integer_divide", "modulo"]:
                    # Operations requiring integers
                    operand1 = safe_int_cast(operand1)
                    operand2 = safe_int_cast(operand2)
                else:
                    # Operations allowing floats
                    operand1 = 0 if operand1 is None else float(operand1)
                    operand2 = 0 if operand2 is None else float(operand2)
            except (ValueError, TypeError):
                # If conversion fails, use safe defaults
                operand1 = (
                    0 if operation in ["random", "integer_divide", "modulo"] else 0.0
                )
                operand2 = (
                    1 if operation in ["random", "integer_divide", "modulo"] else 1.0
                )

            result = None
            if operation == "add":
                result = operand1 + operand2
            elif operation == "subtract":
                result = operand1 - operand2
            elif operation == "multiply":
                result = operand1 * operand2
            elif operation == "divide":
                # Safe division
                if operand2 != 0:
                    result = operand1 / operand2
                else:
                    result = 0
            elif operation == "integer_divide":
                # Safe integer division
                if operand2 != 0:
                    result = operand1 // operand2
                else:
                    result = 0
            elif operation == "modulo":
                # Safe modulo
                result = safe_modulo(operand1, operand2)
            elif operation == "random":
                # Safe random with proper bounds
                min_val = min(operand1, operand2)
                max_val = max(operand1, operand2)
                if min_val == max_val:
                    result = min_val
                else:
                    result = random.randint(min_val, max_val)

            if result is not None and store_key:
                species.memory[str(store_key)] = result

            success = result is not None
            result_data = {"result": result}

        elif op_type == OperationType.BRANCH:
            # Conditional branching
            condition = str(params.get("condition", "true"))
            value1 = params.get("value1", 0)
            value2 = params.get("value2", 0)

            # Ensure values have appropriate types for comparison
            try:
                value1 = 0 if value1 is None else float(value1)
                value2 = 0 if value2 is None else float(value2)
            except (ValueError, TypeError):
                value1, value2 = 0, 0

            condition_met = False
            if condition == "equal":
                condition_met = value1 == value2
            elif condition == "not_equal":
                condition_met = value1 != value2
            elif condition == "greater":
                condition_met = value1 > value2
            elif condition == "less":
                condition_met = value1 < value2
            elif condition == "true":
                condition_met = True
            elif condition == "random":
                condition_met = random.random() < 0.5

            success = True  # Branch operations always succeed
            result_data = {"condition_met": condition_met}

        elif op_type == OperationType.JUMP:
            # Jump to another rule
            try:
                rule_index = safe_int_cast(params.get("rule_index"))
            except (ValueError, TypeError):
                rule_index = 0

            # Rule index bounds checking
            if 0 <= rule_index < len(species.rules):
                result_data = {"jump_to": rule_index}
                success = True
            else:
                result_data = {"invalid_jump": True}

        elif op_type == OperationType.REPRODUCE:
            # Trigger reproduction
            if species.energy >= config.REPRODUCTION_THRESHOLD:
                result_data = {"can_reproduce": True}
                success = True
            else:
                result_data = {"insufficient_energy": True}

        elif op_type == OperationType.SHARE:
            # Share energy with neighboring species
            direction = params.get("direction", Direction.random())

            # Safely get amount value
            try:
                amount = float(params.get("amount", species.energy * 0.1))
            except (ValueError, TypeError):
                amount = species.energy * 0.1

            if not hasattr(species, "position") or species.position is None:
                species.position = Position(0, 0)

            target_pos = get_neighbor_position(species.position, direction)

            # Validate indices before accessing world_grid
            if (
                world_grid is not None
                and 0 <= target_pos.y < len(world_grid)
                and 0 <= target_pos.x < len(world_grid[0])
            ):

                target_cell = world_grid[target_pos.y][target_pos.x]

                if target_cell.occupant:
                    # Limit amount to avoid oversharing
                    transfer_amount = min(amount, species.energy * 0.5)
                    species.energy -= transfer_amount
                    target_cell.occupant.energy += transfer_amount
                    success = True
                    result_data = {"shared": transfer_amount}
                else:
                    result_data = {"no_recipient": True}
            else:
                result_data = {"error": "invalid position or world"}

        elif op_type == OperationType.DEFEND:
            # Set up defenses (increase energy cost for attackers)
            try:
                defense_level = float(params.get("level", 0.5))
                duration = safe_int_cast(params.get("duration", 3))
            except (ValueError, TypeError):
                defense_level = 0.5
                duration = 3

            # Store defense information in memory
            current_tick = species.memory.get("current_tick", 0)
            species.memory["defense"] = {
                "level": min(0.9, defense_level),
                "until_tick": current_tick + duration,
            }

            success = True
            result_data = {"defense_active": True}
            energy_cost *= 1 + defense_level  # Higher defense costs more energy

        elif op_type == OperationType.MUTATE:
            # Self-modification of rules
            try:
                target_rule = safe_int_cast(params.get("rule_index"))
                mutation_type = str(params.get("type", "modify"))
            except (ValueError, TypeError):
                target_rule = 0
                mutation_type = "modify"

            if 0 <= target_rule < len(species.rules):
                if (
                    mutation_type == "duplicate"
                    and len(species.rules) < config.MAX_RULES_PER_SPECIES
                ):
                    # Duplicate a rule
                    new_rule = [inst.clone() for inst in species.rules[target_rule]]
                    species.rules.append(new_rule)
                    species.rule_usage_counts.append(0)
                    success = True
                    result_data = {"duplicated": target_rule}
                    energy_cost *= 3  # Duplication is expensive

                elif mutation_type == "modify":
                    # Modify a single instruction in the rule
                    if species.rules[target_rule]:
                        try:
                            instr_idx = safe_int_cast(
                                params.get(
                                    "instruction_index",
                                    random.randint(
                                        0, len(species.rules[target_rule]) - 1
                                    ),
                                )
                            )
                        except (ValueError, TypeError):
                            instr_idx = 0

                        if 0 <= instr_idx < len(species.rules[target_rule]):
                            # Simple parameter mutation
                            instr = species.rules[target_rule][instr_idx]
                            param_key = random.choice(
                                list(instr.params.keys()) or ["value"]
                            )

                            if isinstance(instr.params.get(param_key, 0), (int, float)):
                                # Adjust numeric parameter by ±20%
                                current = instr.params.get(param_key, 0)
                                adjustment = current * (random.random() * 0.4 - 0.2)
                                instr.params[param_key] = current + adjustment

                            success = True
                            result_data = {
                                "modified": f"rule{target_rule}.{instr_idx}.{param_key}"
                            }
                            energy_cost *= 2
            else:
                result_data = {"invalid_rule": True}

    except Exception as e:
        logger.error(f"Error executing instruction {op_type}: {e}")
        success = False
        result_data = {"error": str(e)}

    # Consume energy for the operation
    species.consume_energy(energy_cost)

    return success, result_data


def run_rule(species: RuleSpecies, rule_index: int) -> Tuple[bool, Optional[int]]:
    """
    Run a specific rule for a species.

    Args:
        species: The species running the rule
        rule_index: Index of the rule to run

    Returns:
        Tuple of (rule_completed, jump_to_rule)
    """
    if rule_index < 0 or rule_index >= len(species.rules):
        return False, None

    rule = species.rules[rule_index]
    species.record_rule_execution(rule_index)

    # Execute each instruction in the rule
    for i, instruction in enumerate(rule):
        success, result = execute_instruction(species, instruction)

        # Handle branching
        if instruction.op_type == OperationType.BRANCH:
            if not result.get("condition_met", False):
                # Skip the next instruction if condition not met
                if i + 1 < len(rule):
                    continue

        # Handle jumps
        if instruction.op_type == OperationType.JUMP and success:
            jump_target = result.get("jump_to")
            if jump_target is not None:
                return True, jump_target

        # Handle failures
        if not success and instruction.op_type not in (
            OperationType.BRANCH,
            OperationType.JUMP,
        ):
            return False, None

        # Check if species died during rule execution
        if species.energy <= 0:
            return False, None

    return True, None


def run_species(species: RuleSpecies) -> bool:
    """
    Run the rules of a species for one simulation tick.

    Args:
        species: The species to run

    Returns:
        True if species is still alive after running
    """
    if not species.rules:
        return False

    # Store current tick in species memory
    if "current_tick" in species.memory:
        species.memory["current_tick"] += 1
    else:
        species.memory["current_tick"] = 0

    # Base energy consumption per tick
    species.consume_energy(config.BASE_ENERGY_CONSUMPTION)

    # Check if species is still alive
    if species.energy <= 0:
        return False

    # Start with a random rule or the last executed rule
    if species.last_executed_rule >= 0 and species.last_executed_rule < len(
        species.rules
    ):
        rule_index = species.last_executed_rule
    else:
        rule_index = random.randint(0, len(species.rules) - 1)

    # Run rules with a cycle limit to prevent infinite loops
    cycles = 0
    while cycles < config.MAX_CYCLES_PER_TICK:
        # Run the current rule
        completed, jump_to = run_rule(species, rule_index)

        # Update rule index for next cycle
        if jump_to is not None:
            rule_index = jump_to
        else:
            rule_index = (rule_index + 1) % len(species.rules)

        cycles += 1

        # Check if species died during execution
        if species.energy <= 0:
            return False

        # Break after a reasonable number of cycles
        if cycles >= min(len(species.rules) * 2, config.MAX_CYCLES_PER_TICK):
            break

    return True
