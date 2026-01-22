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
def generate_getitem(meta_val, fx_node: torch.fx.Node, arg: TensorArgument, idx: int):
    name = arg.name
    individual_output = self.graph.create_node('call_function', operator.getitem, (fx_node, idx), name=name)
    self.sync_fx_node(name, individual_output)
    meta_val.append(self.serialized_name_to_meta[name])
    individual_output.meta.update(deserialized_metadata)