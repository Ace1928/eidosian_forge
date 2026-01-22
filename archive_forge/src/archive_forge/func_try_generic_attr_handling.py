import functools
import inspect
import operator
import types
from typing import Dict, List
import sympy
import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch.fx.experimental.symbolic_shapes import (
from .. import config, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable
def try_generic_attr_handling():
    from .builder import wrap_fx_proxy
    from .misc import GetAttrVariable
    try:
        static_attr = inspect.getattr_static(torch.Tensor, name)
    except AttributeError:
        return None
    if type(static_attr) != types.GetSetDescriptorType:
        return None
    return wrap_fx_proxy(tx=tx, proxy=GetAttrVariable.create_getattr_proxy(self.as_proxy(), name))