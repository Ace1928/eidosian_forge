from __future__ import annotations
from typing import Type
from . import exc as orm_exc
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .. import exc
from .. import inspect
from ..sql import and_
from ..sql import operators
from ..sql.sqltypes import Integer
from ..sql.sqltypes import Numeric
from ..util import warn_deprecated
def visit_is_binary_op(self, operator, eval_left, eval_right, clause):

    def evaluate(obj):
        left_val = eval_left(obj)
        right_val = eval_right(obj)
        if left_val is _EXPIRED_OBJECT or right_val is _EXPIRED_OBJECT:
            return _EXPIRED_OBJECT
        return left_val == right_val
    return evaluate