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
def sym_ite_impl(pred_node, then_node, else_node):
    from torch.fx.experimental.symbolic_shapes import safe_expand
    out_hint = then_node.hint if pred_node.hint else else_node.hint
    if sym_function_mode():
        return to_node(pred_node, handle_sym_dispatch(sym_ite, (wrap_node(pred_node), wrap_node(then_node), wrap_node(else_node)), {}))
    try:
        out = func(pred_node.expr, then_node.expr, else_node.expr)
    except Exception:
        log.warning('failed to eval %s(%s, %s, %s)', method, pred_node.expr, then_node.expr, else_node.expr)
        raise
    out = safe_expand(out)
    fx_node, _ = pred_node.shape_env.create_fx_call_function(sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node))
    return SymNode(out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node)