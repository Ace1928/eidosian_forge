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
def deserialize_inputs(self, target: torch._ops.OpOverload, serialized_node: Node):
    schema_args = target._schema.arguments
    actual_args = {input.name: self.deserialize_input(input.arg) for input in serialized_node.inputs}
    args = []
    kwargs = {}
    for schema_arg in schema_args:
        is_positional = not schema_arg.has_default_value() and (not schema_arg.kwarg_only)
        if is_positional:
            args.append(actual_args[schema_arg.name])
        elif schema_arg.name in actual_args:
            kwargs[schema_arg.name] = actual_args[schema_arg.name]
    return (tuple(args), kwargs)