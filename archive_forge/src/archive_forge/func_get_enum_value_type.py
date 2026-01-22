import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent
from typing import Type
import torch
from torch._C import (
from torch._sources import get_source_lines_and_file
from .._jit_internal import (  # type: ignore[attr-defined]
from ._state import _get_script_class
from torch._ops import OpOverloadPacket
def get_enum_value_type(e: Type[enum.Enum], loc):
    enum_values: List[enum.Enum] = list(e)
    if not enum_values:
        raise ValueError(f"No enum values defined for: '{e.__class__}'")
    types = {type(v.value) for v in enum_values}
    ir_types = [try_ann_to_type(t, loc) for t in types]
    res = torch._C.unify_type_list(ir_types)
    if not res:
        return AnyType.get()
    return res