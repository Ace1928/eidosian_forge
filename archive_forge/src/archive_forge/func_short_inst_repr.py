import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def short_inst_repr(self) -> str:
    return f'Instruction(opname={self.opname}, offset={self.offset})'