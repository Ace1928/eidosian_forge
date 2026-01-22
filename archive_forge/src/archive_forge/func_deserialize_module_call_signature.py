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
def deserialize_module_call_signature(self, module_call_signature: ModuleCallSignature) -> ep.ModuleCallSignature:
    return ep.ModuleCallSignature(inputs=[self.deserialize_argument_spec(x) for x in module_call_signature.inputs], outputs=[self.deserialize_argument_spec(x) for x in module_call_signature.outputs], in_spec=treespec_loads(module_call_signature.in_spec), out_spec=treespec_loads(module_call_signature.out_spec))