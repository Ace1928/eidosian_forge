import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def create_rot_n(n) -> List[Instruction]:
    """
    Returns a "simple" sequence of instructions that rotates TOS to the n-th
    position in the stack. For Python < 3.11, returns a single ROT_*
    instruction. If no such instruction exists, an error is raised and the
    caller is expected to generate an equivalent sequence of instructions.
    For Python >= 3.11, any rotation can be expressed as a simple sequence of
    swaps.
    """
    if n <= 1:
        return []
    if sys.version_info >= (3, 11):
        return [create_instruction('SWAP', arg=i) for i in range(n, 1, -1)]
    if sys.version_info < (3, 8) and n >= 4:
        raise AttributeError(f'rotate {n} not supported for Python < 3.8')
    if sys.version_info < (3, 10) and n >= 5:
        raise AttributeError(f'rotate {n} not supported for Python < 3.10')
    if n <= 4:
        return [create_instruction('ROT_' + ['TWO', 'THREE', 'FOUR'][n - 2])]
    return [create_instruction('ROT_N', arg=n)]