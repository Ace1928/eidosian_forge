from typing import Dict, Optional, Set
import torch
from torch._ops import OpOverload, OpOverloadPacket, HigherOrderOperator
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
def call_operator(self, op, args, kwargs, meta):
    if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
        return super().call_operator(_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op], args, kwargs, meta)
    if op in _BLACK_LISTED_OPS or isinstance(op, HigherOrderOperator):
        return super().call_operator(op, args, kwargs, meta)
    if (view_copy_op := get_view_copy_of_view_op(op._schema)):
        return super().call_operator(view_copy_op, args, kwargs, meta)
    return super().call_operator(op, args, kwargs, meta)