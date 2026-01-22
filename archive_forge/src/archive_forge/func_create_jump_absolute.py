import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def create_jump_absolute(target) -> Instruction:
    inst = 'JUMP_FORWARD' if sys.version_info >= (3, 11) else 'JUMP_ABSOLUTE'
    return create_instruction(inst, target=target)