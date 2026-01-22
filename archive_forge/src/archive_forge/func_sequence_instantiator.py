import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def sequence_instantiator(strings: List[str]) -> Any:
    if isinstance(inner_meta.nargs, int) and len(strings) % inner_meta.nargs != 0:
        raise ValueError(f'input {strings} is of length {len(strings)}, which is not divisible by {inner_meta.nargs}.')
    out = []
    step = inner_meta.nargs if isinstance(inner_meta.nargs, int) else 1
    for i in range(0, len(strings), step):
        out.append(make(strings[i:i + inner_meta.nargs]))
    assert container_type is not None
    return container_type(out)