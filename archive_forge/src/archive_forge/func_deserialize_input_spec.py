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
def deserialize_input_spec(self, i: InputSpec) -> ep.InputSpec:
    if i.user_input is not None:
        return ep.InputSpec(kind=ep.InputKind.USER_INPUT, arg=self.deserialize_argument_spec(i.user_input.arg), target=None)
    elif i.parameter is not None:
        return ep.InputSpec(kind=ep.InputKind.PARAMETER, arg=PyTensorArgument(name=i.parameter.arg.name), target=i.parameter.parameter_name)
    elif i.buffer is not None:
        return ep.InputSpec(kind=ep.InputKind.BUFFER, arg=PyTensorArgument(name=i.buffer.arg.name), target=i.buffer.buffer_name)
    elif i.tensor_constant is not None:
        return ep.InputSpec(kind=ep.InputKind.CONSTANT_TENSOR, arg=PyTensorArgument(name=i.tensor_constant.arg.name), target=i.tensor_constant.tensor_constant_name)
    else:
        raise AssertionError(f'Unkown input spec {i}')