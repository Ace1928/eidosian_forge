import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def merge_programs(prog_list: Sequence[Program]) -> Program:
    """
    Merges a list of pyQuil programs into a single one by appending them in sequence.
    If multiple programs in the list contain the same gate and/or noisy gate definition
    with identical name, this definition will only be applied once. If different definitions
    with the same name appear multiple times in the program list, each will be applied once
    in the order of last occurrence.

    :param prog_list: A list of pyquil programs
    :return: a single pyQuil program
    """
    definitions = [gate for prog in prog_list for gate in Program(prog).defined_gates]
    seen: Dict[str, List[DefGate]] = {}
    for definition in reversed(definitions):
        name = definition.name
        if name in seen.keys():
            if definition not in seen[name]:
                seen[name].append(definition)
        else:
            seen[name] = [definition]
    new_definitions = [gate for key in seen.keys() for gate in reversed(seen[key])]
    p = Program(*[prog._instructions for prog in prog_list])
    for definition in new_definitions:
        if isinstance(definition, DefPermutationGate):
            p.inst(DefPermutationGate(definition.name, list(definition.permutation)))
        else:
            p.defgate(definition.name, definition.matrix, definition.parameters)
    return p