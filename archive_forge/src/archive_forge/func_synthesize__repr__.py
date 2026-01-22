import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def synthesize__repr__(cls) -> ParsedDef:
    return compose_fn(cls, '__repr__', [f"return '{cls.__name__}(" + ', '.join([f'{field.name}=self.{field.name}' for field in dataclasses.fields(cls) if field.repr]) + ")'"], signature='(self) -> str')