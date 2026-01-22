import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def create_call_method(nargs) -> List[Instruction]:
    if sys.version_info >= (3, 11):
        return [create_instruction('PRECALL', arg=nargs), create_instruction('CALL', arg=nargs)]
    return [create_instruction('CALL_METHOD', arg=nargs)]