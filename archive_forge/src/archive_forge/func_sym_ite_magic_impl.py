import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def sym_ite_magic_impl(pred, then_val, else_val):
    pred_node = pred.node
    then_node = to_node(pred_node, then_val)
    else_node = to_node(pred_node, else_val)
    if then_node is NotImplemented or else_node is NotImplemented:
        return NotImplemented
    assert isinstance(then_node, SymNode) and isinstance(else_node, SymNode) and (then_node.pytype == else_node.pytype)
    ret = wrap_node(getattr(pred.node, method_attr)(then_node, else_node))
    return get_constant(ret) if ret.node.is_constant() else ret