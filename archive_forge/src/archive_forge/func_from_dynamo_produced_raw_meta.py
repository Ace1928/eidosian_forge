from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
@classmethod
def from_dynamo_produced_raw_meta(cls, raw_meta: _DYNAMO_NN_MODULE_META_TYPE) -> _ModuleMeta:
    """Create a module meta from raw meta produced by FX dynamo tracer."""
    module_name, (qualified_name, module_class) = raw_meta
    return _ModuleMeta(module_name, module_class, raw_meta)