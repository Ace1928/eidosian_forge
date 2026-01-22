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
@contextmanager
def save_graph_module(self) -> Iterator[None]:
    saved = (self.graph, self.module, self.serialized_name_to_node, self.serialized_name_to_meta)
    self.graph = torch.fx.Graph()
    self.module = torch.nn.Module()
    self.serialized_name_to_node = {}
    self.serialized_name_to_meta = {}
    try:
        yield
    finally:
        self.graph, self.module, self.serialized_name_to_node, self.serialized_name_to_meta = saved