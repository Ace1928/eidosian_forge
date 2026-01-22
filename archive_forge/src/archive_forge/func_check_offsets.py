import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def check_offsets(instructions) -> None:
    offset = 0
    for inst in instructions:
        assert inst.offset == offset
        offset += instruction_size(inst)