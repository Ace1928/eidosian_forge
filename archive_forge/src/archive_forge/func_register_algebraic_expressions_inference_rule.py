from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def register_algebraic_expressions_inference_rule(call_target):

    def register(fn):
        if call_target in _RULES:
            raise RuntimeError(f'Rule already registered for {call_target}!')
        _RULES[call_target] = fn
        return fn
    return register