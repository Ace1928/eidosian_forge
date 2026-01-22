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
def percolate_declares(program: Program) -> Program:
    """
    Move all the DECLARE statements to the top of the program. Return a fresh object.

    :param program: Perhaps jumbled program.
    :return: Program with DECLAREs all at the top and otherwise the same sorted contents.
    """
    declare_program = Program()
    instrs_program = Program()
    for instr in program:
        if isinstance(instr, Declare):
            declare_program += instr
        else:
            instrs_program += instr
    p = declare_program + instrs_program
    p._defined_gates = program._defined_gates
    return p