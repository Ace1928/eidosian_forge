import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def maybe_pop_n(n):
    for _ in range(n):
        if output and output[-1].opcode == dis.EXTENDED_ARG:
            output.pop()