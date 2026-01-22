import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
def reconstruct_getitem(source: Union['GetItemSource', 'ODictGetItemSource'], codegen, index_is_slice):
    instrs = source.base.reconstruct(codegen)
    if isinstance(source.index, Source):
        instrs.extend(source.index.reconstruct(codegen))
    elif index_is_slice:
        assert isinstance(source, GetItemSource)
        instrs.append(codegen.create_load_const(source.unpack_slice()))
    else:
        instrs.append(codegen.create_load_const(source.index))
    return instrs