"""
Instruction set implementation for the ECosmos system.

This module defines functions for each operation code (opcode) that can be
executed by the interpreter. These functions manipulate the RuleSpecies
according to the rule's operands and modifiers.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import random
import math
import logging

from data_structures import Rule, RuleSpecies, Position
import config

# Set up logging
logger = logging.getLogger(__name__)


def safe_cast(val: Any, to_type: type, default: Any) -> Any:
    """Safely cast a value to a specified type."""
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def get_operand_value(species: RuleSpecies, operand: Any) -> Any:
    """
    Handle operands that might be references to memory.
    If the operand is a string starting with '$', treat it as a memory reference.
    """
    if isinstance(operand, str) and operand.startswith("$"):
        return species.execution_memory.load(operand[1:])
    return operand


def execute_nop(species: RuleSpecies, rule: Rule) -> None:
    """No operation - does nothing but consumes a cycle."""
    pass


def execute_arithmetic(species: RuleSpecies, rule: Rule, opcode: str) -> None:
    """
    Execute arithmetic operations (ADD, SUB, MUL, DIV, MOD).

    Operands:
        [0]: First value or memory reference
        [1]: Second value or memory reference
        [2]: (Optional) Output memory location
    """
    if len(rule.operands) < 2:
        return

    # Get actual values from operands
    a = get_operand_value(species, rule.operands[0])
    b = get_operand_value(species, rule.operands[1])

    # Ensure numeric values
    a = safe_cast(a, float, 0.0)
    b = safe_cast(b, float, 0.0)

    result = 0.0
    if opcode == "ADD":
        result = a + b
    elif opcode == "SUB":
        result = a - b
    elif opcode == "MUL":
        result = a * b
    elif opcode == "DIV":
        result = a / b if b != 0 else 0
    elif opcode == "MOD":
        result = a % b if b != 0 else 0

    # Store result
    if (
        len(rule.operands) > 2
        and isinstance(rule.operands[2], str)
        and rule.operands[2].startswith("$")
    ):
        # Store in specified memory location
        species.execution_memory.store(rule.operands[2][1:], result)
    else:
        # Replace first operand with result
        rule.operands[0] = result


def execute_logic(species: RuleSpecies, rule: Rule, opcode: str) -> None:
    """
    Execute logical operations (XOR, AND, OR).

    Operands:
        [0]: First value or memory reference
        [1]: Second value or memory reference
        [2]: (Optional) Output memory location
    """
    if len(rule.operands) < 2:
        return

    # Get actual values
    a = get_operand_value(species, rule.operands[0])
    b = get_operand_value(species, rule.operands[1])

    # Convert to booleans
    a = bool(a)
    b = bool(b)

    result = False
    if opcode == "XOR":
        result = a ^ b
    elif opcode == "AND":
        result = a and b
    elif opcode == "OR":
        result = a or b

    # Store result
    if (
        len(rule.operands) > 2
        and isinstance(rule.operands[2], str)
        and rule.operands[2].startswith("$")
    ):
        species.execution_memory.store(rule.operands[2][1:], result)
    else:
        rule.operands[0] = result


def execute_not(species: RuleSpecies, rule: Rule) -> None:
    """
    Execute logical NOT operation.

    Operands:
        [0]: Value to negate or memory reference
        [1]: (Optional) Output memory location
    """
    if len(rule.operands) < 1:
        return

    value = get_operand_value(species, rule.operands[0])
    result = not bool(value)

    if (
        len(rule.operands) > 1
        and isinstance(rule.operands[1], str)
        and rule.operands[1].startswith("$")
    ):
        species.execution_memory.store(rule.operands[1][1:], result)
    else:
        rule.operands[0] = result


def execute_shift(species: RuleSpecies, rule: Rule, direction: str) -> None:
    """
    Execute bit shift operations (SHIFT_LEFT, SHIFT_RIGHT).

    Operands:
        [0]: Value to shift or memory reference
        [1]: Number of bits to shift or memory reference
        [2]: (Optional) Output memory location
    """
    if len(rule.operands) < 2:
        return

    value = get_operand_value(species, rule.operands[0])
    shift_amount = get_operand_value(species, rule.operands[1])

    # Ensure proper types
    value = safe_cast(value, int, 0)
    shift_amount = safe_cast(shift_amount, int, 0)

    # Bound shift amount to prevent excessive shifts
    shift_amount = max(0, min(shift_amount, 31))

    if direction == "LEFT":
        result = value << shift_amount
    else:  # RIGHT
        result = value >> shift_amount

    if (
        len(rule.operands) > 2
        and isinstance(rule.operands[2], str)
        and rule.operands[2].startswith("$")
    ):
        species.execution_memory.store(rule.operands[2][1:], result)
    else:
        rule.operands[0] = result


def execute_jump(species: RuleSpecies, rule: Rule, condition: str) -> int:
    """
    Execute jump operations (JUMP, JUMP_IF, JUMP_UNLESS).
    Returns the jump offset to apply to the instruction pointer.

    Operands:
        [0]: Jump offset or target or condition (for conditional jumps)
        [1]: Jump offset or target (for conditional jumps)

    Returns:
        int: Jump offset to apply
    """
    if condition == "":  # Unconditional jump
        if len(rule.operands) < 1:
            return 0
        offset = get_operand_value(species, rule.operands[0])
        return safe_cast(offset, int, 0)

    elif condition in ["IF", "UNLESS"]:
        if len(rule.operands) < 2:
            return 0

        test_value = get_operand_value(species, rule.operands[0])
        offset = get_operand_value(species, rule.operands[1])

        # Convert to proper types
        test_value = bool(test_value)
        offset = safe_cast(offset, int, 0)

        if (condition == "IF" and test_value) or (
            condition == "UNLESS" and not test_value
        ):
            return offset

    return 0  # Default: no jump


def execute_stack(species: RuleSpecies, rule: Rule, operation: str) -> None:
    """
    Execute stack operations (STACK_PUSH, STACK_POP).

    For STACK_PUSH:
        Operands[0]: Value to push or memory reference

    For STACK_POP:
        Operands[0]: (Optional) Memory location to store popped value
    """
    stack = species.execution_memory.stack

    if operation == "PUSH":
        if len(rule.operands) < 1:
            return
        value = get_operand_value(species, rule.operands[0])
        stack.push(value)

    elif operation == "POP":
        value = stack.pop()
        if value is not None and len(rule.operands) > 0:
            if isinstance(rule.operands[0], str) and rule.operands[0].startswith("$"):
                species.execution_memory.store(rule.operands[0][1:], value)
            else:
                rule.operands[0] = value


def execute_memory(species: RuleSpecies, rule: Rule, operation: str) -> None:
    """
    Execute memory operations (STORE, LOAD).

    For STORE:
        Operands[0]: Memory key (without '$')
        Operands[1]: Value to store

    For LOAD:
        Operands[0]: Memory key (without '$')
        Operands[1]: (Optional) Where to place the loaded value
    """
    if len(rule.operands) < 2:
        return

    if operation == "STORE":
        key = str(rule.operands[0])
        value = get_operand_value(species, rule.operands[1])
        species.execution_memory.store(key, value)

    elif operation == "LOAD":
        key = str(rule.operands[0])
        value = species.execution_memory.load(key)

        if len(rule.operands) > 1:
            if isinstance(rule.operands[1], str) and rule.operands[1].startswith("$"):
                # Store in another memory location
                target_key = rule.operands[1][1:]
                species.execution_memory.store(target_key, value)
            else:
                # Store in operand
                rule.operands[1] = value


def execute_copy(species: RuleSpecies, rule: Rule) -> None:
    """
    Copy a rule within the species.

    Operands:
        [0]: (Optional) Index of rule to copy, or random if not specified
    """
    if len(species.rules) == 0:
        return

    # Don't exceed maximum rules limit
    if len(species.rules) >= config.MAX_RULES_PER_SPECIES:
        return

    # Determine which rule to copy
    if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
        idx = rule.operands[0] % len(species.rules)
    else:
        idx = random.randint(0, len(species.rules) - 1)

    original_rule = species.rules[idx]
    cloned_rule = original_rule.clone()
    species.rules.append(cloned_rule)


def execute_mutate(species: RuleSpecies, rule: Rule) -> None:
    """
    Mutate a random rule in the species.

    Operands:
        [0]: (Optional) Index of rule to mutate, or random if not specified
        [1]: (Optional) Mutation strength (0.0-1.0)
    """
    if len(species.rules) == 0:
        return

    # Determine target rule
    if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
        idx = rule.operands[0] % len(species.rules)
        target_rule = species.rules[idx]
    else:
        target_rule = random.choice(species.rules)

    # Determine mutation strength
    strength = 1.0
    if len(rule.operands) > 1:
        strength = safe_cast(rule.operands[1], float, 1.0)
        strength = max(0.1, min(strength, 1.0))  # Bound between 0.1 and 1.0

    # Choose what to mutate
    mutation_type = random.choices(
        ["opcode", "operand", "add_operand", "remove_operand", "modifier"],
        weights=[0.2, 0.4, 0.2, 0.1, 0.1],
    )[0]

    if mutation_type == "opcode":
        possible_opcodes = list(config.INSTRUCTION_COSTS.keys())
        target_rule.opcode = random.choice(possible_opcodes)

    elif mutation_type == "operand" and target_rule.operands:
        operand_idx = random.randint(0, len(target_rule.operands) - 1)
        val = target_rule.operands[operand_idx]

        if isinstance(val, int):
            # Mutate integer values with scaled changes
            change = int(random.gauss(0, 5 * strength))
            target_rule.operands[operand_idx] = val + change

        elif isinstance(val, float):
            # Mutate float values
            change = random.gauss(0, 0.5 * strength)
            target_rule.operands[operand_idx] = val + change

        elif isinstance(val, bool):
            # Flip boolean values
            target_rule.operands[operand_idx] = not val

        elif isinstance(val, str):
            # Potentially mutate memory references
            if random.random() < 0.5:
                # Change memory location
                if val.startswith("$"):
                    new_key = f"${random.randint(0, 9)}"
                    target_rule.operands[operand_idx] = new_key
            else:
                # Convert to a different type
                target_rule.operands[operand_idx] = random.randint(-10, 10)

        else:
            # Replace with a random value
            target_rule.operands[operand_idx] = random.randint(-10, 10)

    elif (
        mutation_type == "add_operand" and len(target_rule.operands) < 5
    ):  # Limit operands
        # Add a new operand
        new_type = random.choice(["int", "float", "bool", "memory"])
        if new_type == "int":
            target_rule.operands.append(random.randint(-10, 10))
        elif new_type == "float":
            target_rule.operands.append(random.uniform(-1.0, 1.0))
        elif new_type == "bool":
            target_rule.operands.append(random.choice([True, False]))
        else:  # memory
            target_rule.operands.append(f"${random.randint(0, 9)}")

    elif mutation_type == "remove_operand" and len(target_rule.operands) > 0:
        # Remove a random operand
        operand_idx = random.randint(0, len(target_rule.operands) - 1)
        target_rule.operands.pop(operand_idx)

    elif mutation_type == "modifier":
        # Mutate a modifier or add a new one
        mod_key = random.choice(["priority", "condition", "repeat"])
        if mod_key in target_rule.modifiers or random.random() < 0.5:
            # Modify existing or add new
            target_rule.modifiers[mod_key] = random.randint(1, 5)


def execute_recombine(species: RuleSpecies, rule: Rule) -> None:
    """
    Combine two rules to create a new hybrid rule.

    Operands:
        [0]: (Optional) First parent rule index
        [1]: (Optional) Second parent rule index
    """
    if len(species.rules) < 2:
        return

    # Don't exceed maximum rules
    if len(species.rules) >= config.MAX_RULES_PER_SPECIES:
        return

    # Select parent rules
    if (
        len(rule.operands) >= 2
        and isinstance(rule.operands[0], int)
        and isinstance(rule.operands[1], int)
    ):
        idx1 = rule.operands[0] % len(species.rules)
        idx2 = rule.operands[1] % len(species.rules)
        r1 = species.rules[idx1]
        r2 = species.rules[idx2]
    else:
        # Randomly select parents
        r1, r2 = random.sample(species.rules, 2)

    # Create hybrid rule
    new_opcode = random.choice([r1.opcode, r2.opcode])
    new_operands = []

    # Mix operands from both parents
    pivot = (
        min(len(r1.operands), len(r2.operands)) // 2
        if min(len(r1.operands), len(r2.operands)) > 1
        else 1
    )
    new_operands.extend(r1.operands[:pivot])
    new_operands.extend(r2.operands[pivot:])

    # Mix modifiers
    new_modifiers = {**r1.modifiers, **r2.modifiers}

    # Create new rule with mixed properties
    new_rule = Rule(
        opcode=new_opcode,
        operands=new_operands,
        modifiers=new_modifiers,
        execution_cost=(r1.execution_cost + r2.execution_cost) / 2,
        lifecycle=(r1.lifecycle + r2.lifecycle) // 2,
    )

    species.rules.append(new_rule)


def execute_erase(species: RuleSpecies, rule: Rule) -> None:
    """
    Remove a rule from the species.

    Operands:
        [0]: (Optional) Index of rule to erase, or random if not specified
    """
    if len(species.rules) <= 1:  # Preserve at least one rule
        return

    # Select rule to erase
    if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
        idx = rule.operands[0] % len(species.rules)
    else:
        idx = random.randint(0, len(species.rules) - 1)

    # Don't allow erasing the current rule during execution
    if rule in species.rules and species.rules.index(rule) == idx:
        return

    species.rules.pop(idx)


def execute_reflect(species: RuleSpecies, rule: Rule) -> None:
    """
    Self-inspection: examine and potentially modify rules based on their properties.

    Operands:
        [0]: (Optional) Property to reflect on ('opcode', 'operands', etc.)
        [1]: (Optional) Memory location to store reflection result
    """
    reflection_type = "count"
    if len(rule.operands) > 0 and isinstance(rule.operands[0], str):
        reflection_type = rule.operands[0]

    result = None

    if reflection_type == "count":
        result = len(species.rules)
    elif reflection_type == "opcodes":
        result = [r.opcode for r in species.rules]
    elif reflection_type == "energy":
        result = species.energy
    elif reflection_type == "usage":
        result = (species.cpu_usage, species.ram_usage)

    # Store result in memory if specified
    if (
        len(rule.operands) > 1
        and isinstance(rule.operands[1], str)
        and rule.operands[1].startswith("$")
    ):
        species.execution_memory.store(rule.operands[1][1:], result)

    # Mark rules as reflected
    for r in species.rules:
        r.modifiers["reflected"] = True


def execute_interpret(species: RuleSpecies, rule: Rule) -> None:
    """
    Interpret operand(s) as new rule(s) or instructions.

    Operands:
        [0]: Data to interpret as a rule or instructions
    """
    if len(rule.operands) < 1:
        return

    # Don't exceed maximum rules
    if len(species.rules) >= config.MAX_RULES_PER_SPECIES:
        return

    data = get_operand_value(species, rule.operands[0])

    # Try to interpret as a rule definition
    if isinstance(data, dict) and "opcode" in data:
        try:
            new_opcode = data["opcode"]
            if (
                not isinstance(new_opcode, str)
                or new_opcode not in config.INSTRUCTION_COSTS
            ):
                return

            new_operands = data.get("operands", [])
            new_modifiers = data.get("modifiers", {})

            new_rule = Rule(
                opcode=new_opcode,
                operands=new_operands,
                modifiers=new_modifiers,
                execution_cost=config.INSTRUCTION_COSTS.get(new_opcode, 5),
                lifecycle=config.DEFAULT_RULE_LIFECYCLE,
            )
            species.rules.append(new_rule)
        except (TypeError, ValueError):
            # Invalid format, ignore
            pass
    elif isinstance(data, str):
        # Try to interpret string as opcode
        if data in config.INSTRUCTION_COSTS:
            new_rule = Rule(
                opcode=data,
                operands=[],
                execution_cost=config.INSTRUCTION_COSTS[data],
                lifecycle=config.DEFAULT_RULE_LIFECYCLE,
            )
            species.rules.append(new_rule)


def execute_encode_decode(species: RuleSpecies, rule: Rule, operation: str) -> None:
    """
    Encode or decode rules for compression or transformation.

    Operands:
        [0]: (Optional) Index of rule to encode/decode, or all if not specified
        [1]: (Optional) Memory location to store encoded/decoded data
    """
    if operation == "ENCODE":
        # Mark rules as encoded/compressed
        for r in species.rules:
            r.modifiers["encoded"] = True

        # Store a simplified representation if requested
        if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
            idx = rule.operands[0] % len(species.rules)
            target_rule = species.rules[idx]

            # Create a simplified representation
            encoded = {"op": target_rule.opcode, "data": target_rule.operands}

            if (
                len(rule.operands) > 1
                and isinstance(rule.operands[1], str)
                and rule.operands[1].startswith("$")
            ):
                species.execution_memory.store(rule.operands[1][1:], encoded)

    elif operation == "DECODE":
        # Remove encoded flag
        for r in species.rules:
            if "encoded" in r.modifiers:
                del r.modifiers["encoded"]

        # Try to decode data from memory if specified
        if (
            len(rule.operands) > 0
            and isinstance(rule.operands[0], str)
            and rule.operands[0].startswith("$")
        ):
            encoded = species.execution_memory.load(rule.operands[0][1:])

            if isinstance(encoded, dict) and "op" in encoded and "data" in encoded:
                try:
                    # Don't exceed maximum rules
                    if len(species.rules) >= config.MAX_RULES_PER_SPECIES:
                        return

                    new_rule = Rule(
                        opcode=encoded["op"],
                        operands=encoded["data"],
                        execution_cost=config.INSTRUCTION_COSTS.get(encoded["op"], 5),
                        lifecycle=config.DEFAULT_RULE_LIFECYCLE,
                    )
                    species.rules.append(new_rule)
                except (TypeError, ValueError):
                    # Invalid format, ignore
                    pass


def execute_suppress(species: RuleSpecies, rule: Rule) -> None:
    """
    Temporarily disable or reduce effectiveness of rules.

    Operands:
        [0]: (Optional) Index of rule to suppress, or random if not specified
        [1]: (Optional) Suppression strength (1-5)
    """
    if len(species.rules) == 0:
        return

    # Select rule to suppress
    if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
        idx = rule.operands[0] % len(species.rules)
        target_rule = species.rules[idx]
    else:
        target_rule = random.choice(species.rules)

    # Determine suppression strength
    strength = 1
    if len(rule.operands) > 1 and isinstance(rule.operands[1], int):
        strength = max(1, min(5, rule.operands[1]))

    # Reduce lifecycle
    target_rule.lifecycle = max(1, target_rule.lifecycle - strength)

    # Mark as suppressed
    target_rule.modifiers["suppressed"] = True


def execute_amplify(species: RuleSpecies, rule: Rule) -> None:
    """
    Enhance effectiveness of rules.

    Operands:
        [0]: (Optional) Index of rule to amplify, or random if not specified
        [1]: (Optional) Amplification strength (1-5)
    """
    if len(species.rules) == 0:
        return

    # Select rule to amplify
    if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
        idx = rule.operands[0] % len(species.rules)
        target_rule = species.rules[idx]
    else:
        target_rule = random.choice(species.rules)

    # Determine amplification strength
    strength = 1
    if len(rule.operands) > 1 and isinstance(rule.operands[1], int):
        strength = max(1, min(5, rule.operands[1]))

    # Increase lifecycle
    target_rule.lifecycle += strength

    # Mark as amplified
    target_rule.modifiers["amplified"] = True


def execute_merge(species: RuleSpecies, rule: Rule) -> None:
    """
    Merge two rules into one.

    Operands:
        [0]: (Optional) Index of first rule to merge
        [1]: (Optional) Index of second rule to merge
    """
    if len(species.rules) < 2:
        return

    # Select rules to merge
    if (
        len(rule.operands) >= 2
        and isinstance(rule.operands[0], int)
        and isinstance(rule.operands[1], int)
    ):
        idx1 = rule.operands[0] % len(species.rules)
        idx2 = rule.operands[1] % len(species.rules)
        r1 = species.rules[idx1]
        r2 = species.rules[idx2]
    else:
        # Randomly select rules
        r1, r2 = random.sample(species.rules, 2)

    # Don't merge with self
    if r1 is r2:
        return

    # Combine operands (with deduplication for strings)
    combined_operands = list(r1.operands)
    for op in r2.operands:
        if isinstance(op, str) and op in combined_operands:
            continue
        combined_operands.append(op)

    # Combine modifiers
    combined_modifiers = {**r1.modifiers, **r2.modifiers}

    # Use the more complex opcode
    complex_opcode = (
        r1.opcode
        if config.INSTRUCTION_COSTS.get(r1.opcode, 0)
        >= config.INSTRUCTION_COSTS.get(r2.opcode, 0)
        else r2.opcode
    )

    # Create merged rule
    r1.opcode = complex_opcode
    r1.operands = combined_operands[:5]  # Limit operands to 5
    r1.modifiers = combined_modifiers
    r1.lifecycle = max(r1.lifecycle, r2.lifecycle)

    # Remove second rule
    if r2 in species.rules:
        species.rules.remove(r2)


def execute_split(species: RuleSpecies, rule: Rule) -> None:
    """
    Split a rule into two separate rules.

    Operands:
        [0]: (Optional) Index of rule to split
    """
    if len(species.rules) == 0:
        return

    # Don't exceed maximum rules
    if len(species.rules) >= config.MAX_RULES_PER_SPECIES:
        return

    # Select rule to split
    if len(rule.operands) > 0 and isinstance(rule.operands[0], int):
        idx = rule.operands[0] % len(species.rules)
        target_rule = species.rules[idx]
    else:
        target_rule = random.choice(species.rules)

    # Need at least 2 operands to meaningfully split
    if len(target_rule.operands) < 2:
        return

    # Determine split point
    split_point = len(target_rule.operands) // 2

    # Create two new rules from split
    new_rule1 = Rule(
        opcode=target_rule.opcode,
        operands=target_rule.operands[:split_point],
        modifiers=dict(target_rule.modifiers),
        execution_cost=target_rule.execution_cost,
        lifecycle=target_rule.lifecycle,
    )

    new_rule2 = Rule(
        opcode=target_rule.opcode,
        operands=target_rule.operands[split_point:],
        modifiers=dict(target_rule.modifiers),
        execution_cost=target_rule.execution_cost,
        lifecycle=target_rule.lifecycle,
    )

    # Replace original with first part and add second part
    species.rules[species.rules.index(target_rule)] = new_rule1
    species.rules.append(new_rule2)


def execute_energy_transfer(species: RuleSpecies, rule: Rule) -> None:
    """
    Transfer energy between species or to the environment.

    Operands:
        [0]: (Optional) Direction: -1 for donate, 1 for absorb, 0 for random
        [1]: (Optional) Amount of energy to transfer
    """
    # Determine transfer direction
    direction = 0
    if len(rule.operands) > 0:
        direction = safe_cast(get_operand_value(species, rule.operands[0]), int, 0)

    if direction == 0:
        direction = random.choice([-1, 1])

    # Determine transfer amount
    amount = species.energy * config.ENERGY_TRANSFER_FRACTION
    if len(rule.operands) > 1:
        specified_amount = safe_cast(
            get_operand_value(species, rule.operands[1]), float, 0.0
        )
        if specified_amount > 0:
            amount = min(specified_amount, species.energy * 0.5)  # Cap at 50% of energy

    if direction < 0:  # Donate energy
        species.energy = max(0, species.energy - amount)
        # In a multi-species ecosystem, this could be given to another entity
    else:  # Absorb energy
        # In a real ecosystem, this would come from the environment or other species
        # For now, we'll just add a small random amount
        species.energy += amount * 0.5  # Only get 50% of what you try to absorb


def execute_sense(species: RuleSpecies, rule: Rule) -> None:
    """
    Sense environmental conditions or neighbors.

    Operands:
        [0]: What to sense ('energy', 'neighbors', 'env')
        [1]: (Optional) Memory location to store sensed data
    """
    if len(rule.operands) < 1:
        return

    sense_type = get_operand_value(species, rule.operands[0])
    result = None

    if isinstance(sense_type, str):
        if sense_type == "energy":
            result = species.energy
        elif sense_type == "neighbors":
            # In a full implementation, this would check actual neighbors
            # For now, return a random number of neighbors
            result = random.randint(0, 4)  # 0-4 neighbors
        elif sense_type == "env":
            # In a full implementation, this would return actual environmental factors
            # For now, return random values
            result = {"temperature": random.random(), "resources": random.random()}

    # Store result if requested
    if (
        len(rule.operands) > 1
        and isinstance(rule.operands[1], str)
        and rule.operands[1].startswith("$")
    ):
        species.execution_memory.store(rule.operands[1][1:], result)


def execute_move(species: RuleSpecies, rule: Rule) -> None:
    """
    Request movement in the world.

    Operands:
        [0]: Direction (0-3 for N,E,S,W)
        [1]: (Optional) Memory location to store success result
    """
    # This would be fully implemented in the world update logic
    # For now, just note that movement was requested
    if len(rule.operands) < 1:
        return

    direction = safe_cast(get_operand_value(species, rule.operands[0]), int, 0) % 4

    # Store this request for processing during world update
    species.modifiers["move_request"] = direction

    # In a full implementation, the world update would check this modifier and try to move the species
    # For now, just report success
    if (
        len(rule.operands) > 1
        and isinstance(rule.operands[1], str)
        and rule.operands[1].startswith("$")
    ):
        # Randomly succeed or fail
        species.execution_memory.store(
            rule.operands[1][1:], random.choice([True, False])
        )


def execute_eval(species: RuleSpecies, rule: Rule) -> None:
    """
    High-level meta-execution that treats operands as executable code.

    Operands:
        [0]: Code or operation to evaluate
        [1]: (Optional) Memory location to store result
    """
    if len(rule.operands) < 1:
        return

    eval_code = get_operand_value(species, rule.operands[0])
    result = None

    # Very limited eval for safety - only allow simple arithmetic
    if (
        isinstance(eval_code, str)
        and eval_code.count("+")
        + eval_code.count("-")
        + eval_code.count("*")
        + eval_code.count("/")
        == 1
    ):
        try:
            # Extract numbers
            if "+" in eval_code:
                a, b = eval_code.split("+")
                a, b = float(a.strip()), float(b.strip())
                result = a + b
            elif "-" in eval_code:
                a, b = eval_code.split("-")
                a, b = float(a.strip()), float(b.strip())
                result = a - b
            elif "*" in eval_code:
                a, b = eval_code.split("*")
                a, b = float(a.strip()), float(b.strip())
                result = a * b
            elif "/" in eval_code:
                a, b = eval_code.split("/")
                a, b = float(a.strip()), float(b.strip())
                result = a / b if b != 0 else 0
        except (ValueError, ZeroDivisionError):
            result = None

    # Store result if requested
    if (
        result is not None
        and len(rule.operands) > 1
        and isinstance(rule.operands[1], str)
        and rule.operands[1].startswith("$")
    ):
        species.execution_memory.store(rule.operands[1][1:], result)


def execute_meta_mutate(species: RuleSpecies, rule: Rule) -> None:
    """
    Mutate the mutation parameters themselves.

    Operands:
        [0]: Parameter to mutate ('mutation_rate', 'recombination_rate')
        [1]: (Optional) New value or adjustment
    """
    if len(rule.operands) < 1:
        return

    param = get_operand_value(species, rule.operands[0])

    if not isinstance(param, str):
        return

    # Determine adjustment value
    adjustment = random.uniform(0.9, 1.1)  # Default: scale by 0.9-1.1
    if len(rule.operands) > 1:
        adj_value = get_operand_value(species, rule.operands[1])
        if isinstance(adj_value, (int, float)):
            adjustment = adj_value

    # Apply mutation to global parameters
    # Note: In a real implementation, these might be species-specific parameters
    if param == "mutation_rate":
        species.modifiers["mutation_rate_factor"] = adjustment
    elif param == "recombination_rate":
        species.modifiers["recombination_rate_factor"] = adjustment


def execute_recurse(species: RuleSpecies, rule: Rule) -> None:
    """
    Create a recursive pattern by duplicating rules with modifications.

    Operands:
        [0]: (Optional) Recursion depth (1-3)
        [1]: (Optional) Rule index to recurse on, or random if not specified
    """
    # Determine depth
    depth = 1
    if len(rule.operands) > 0:
        depth = safe_cast(get_operand_value(species, rule.operands[0]), int, 1)
        depth = max(1, min(3, depth))  # Limit recursion depth for safety

    # Don't exceed maximum rules
    if len(species.rules) + depth > config.MAX_RULES_PER_SPECIES:
        return

    # Select rule to recurse on
    if len(rule.operands) > 1 and isinstance(rule.operands[1], int):
        idx = rule.operands[1] % len(species.rules)
        target_rule = species.rules[idx]
    else:
        target_rule = random.choice(species.rules)

    # Create recursive pattern
    for i in range(depth):
        new_rule = target_rule.clone()

        # Add a recursive modifier
        new_rule.modifiers["recursive_depth"] = i + 1

        # Slightly mutate operands if there are any
        if new_rule.operands:
            for j in range(len(new_rule.operands)):
                if isinstance(new_rule.operands[j], (int, float)):
                    new_rule.operands[j] += i + 1  # Increment by recursion level

        species.rules.append(new_rule)


def execute_halt(species: RuleSpecies, rule: Rule) -> None:
    """
    Halt execution for this cycle.

    Operands: None
    """
    # This is implemented in the interpreter by checking for this opcode
    pass


def execute_resource_cost(species: RuleSpecies, rule: Rule) -> None:
    """
    Special opcode to handle resource accounting.

    This is a meta-instruction that doesn't do anything directly but represents
    a cost applied to the species.

    Operands:
        [0]: (Optional) Resource type ("cpu" or "ram")
        [1]: (Optional) Amount to consume
    """
    # This is just a placeholder for the resource cost accounting
    # The actual costs are applied in the interpreter
    resource_type = "generic"
    amount = 1

    if len(rule.operands) > 0:
        resource_type = str(rule.operands[0]).lower()

    if len(rule.operands) > 1:
        amount = safe_cast(rule.operands[1], int, 1)

    # Log the resource usage
    logger.debug(f"Resource cost: {amount} {resource_type}")


def execute_timeout(species: RuleSpecies, rule: Rule) -> None:
    """
    Special opcode to handle execution timeout.

    This is a meta-instruction that represents a timeout in the execution.

    Operands:
        [0]: (Optional) Custom timeout message
    """
    # This is just a placeholder for the timeout handling
    # The actual timeout is enforced by the interpreter
    message = "Execution timed out"

    if len(rule.operands) > 0:
        message = str(rule.operands[0])

    # Log the timeout event
    logger.debug(f"Timeout: {message}")
