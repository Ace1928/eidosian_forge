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
def deserialize_range_constraints(self, symbol_name_to_range: Dict[str, symbolic_shapes.ValueRanges], symbol_name_to_symbol: Dict[str, sympy.Symbol]) -> Dict[sympy.Symbol, ValueRanges]:
    range_constraints = {}
    for k, v in symbol_name_to_range.items():
        if (symbol := symbol_name_to_symbol.get(k)):
            range_constraints[symbol] = v
        else:
            log.warning(f'Symbol {k} did not appear in the graph that was deserialized')
    return range_constraints