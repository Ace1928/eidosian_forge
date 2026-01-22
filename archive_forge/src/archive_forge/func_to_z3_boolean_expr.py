import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
def to_z3_boolean_expr(self, e: sympy.Basic) -> z3.BoolRef:
    z3expr = SympyToZ3(self).run(e)
    assert isinstance(z3expr, z3.BoolRef), f'expected boolean expression. Got: {z3expr}'
    return z3expr