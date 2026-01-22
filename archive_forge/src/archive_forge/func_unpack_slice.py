import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
def unpack_slice(self):
    assert self.index_is_slice
    slice_class, slice_args = self.index
    return slice_class(*slice_args)