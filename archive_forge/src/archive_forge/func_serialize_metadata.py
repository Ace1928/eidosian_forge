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
def serialize_metadata(self, node: torch.fx.Node) -> Dict[str, str]:
    ret = {}
    if (stack_trace := node.meta.get('stack_trace')):
        ret['stack_trace'] = stack_trace
    if (nn_module_stack := node.meta.get('nn_module_stack')):

        def export_nn_module_stack(val):
            assert isinstance(val, tuple) and len(val) == 2
            path, ty = val
            assert isinstance(path, str)
            normalized_ty = ty.__module__ + '.' + ty.__qualname__
            return path + ',' + normalized_ty
        nn_module_list = [f'{k},{export_nn_module_stack(v)}' for k, v in nn_module_stack.items()]
        ret['nn_module_stack'] = ST_DELIMITER.join(nn_module_list)
    if (source_fn_st := node.meta.get('source_fn_stack')):
        source_fn_list = [f'{source_fn[0]},{self.serialize_operator(source_fn[1])}' for source_fn in source_fn_st]
        ret['source_fn_stack'] = ST_DELIMITER.join(source_fn_list)
    return ret