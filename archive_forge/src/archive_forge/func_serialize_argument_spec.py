import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def serialize_argument_spec(self, x: ep.ArgumentSpec) -> Argument:
    if isinstance(x, PyTensorArgument):
        return Argument.create(as_tensor=TensorArgument(name=x.name))
    elif isinstance(x, PySymIntArgument):
        return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.name))
    elif isinstance(x, PyConstantArgument):
        return self.serialize_input(x.value)
    else:
        raise AssertionError('TODO')