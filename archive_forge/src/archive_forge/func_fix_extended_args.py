import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def fix_extended_args(instructions: List[Instruction]) -> int:
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output: List[Instruction] = []

    def maybe_pop_n(n):
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()
    for inst in instructions:
        if inst.opcode == dis.EXTENDED_ARG:
            inst.arg = 0
        elif inst.arg and inst.arg > 16777215:
            maybe_pop_n(3)
            output.append(create_instruction('EXTENDED_ARG', arg=inst.arg >> 24))
            output.append(create_instruction('EXTENDED_ARG', arg=inst.arg >> 16))
            output.append(create_instruction('EXTENDED_ARG', arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 65535:
            maybe_pop_n(2)
            output.append(create_instruction('EXTENDED_ARG', arg=inst.arg >> 16))
            output.append(create_instruction('EXTENDED_ARG', arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 255:
            maybe_pop_n(1)
            output.append(create_instruction('EXTENDED_ARG', arg=inst.arg >> 8))
        output.append(inst)
    added = len(output) - len(instructions)
    assert added >= 0
    instructions[:] = output
    return added