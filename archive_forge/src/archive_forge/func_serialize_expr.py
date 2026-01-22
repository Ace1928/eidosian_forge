import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_expr(self, expr):
    """
        Serialize ``BaseExpr`` based expression into a dictionary.

        Parameters
        ----------
        expr : BaseExpr
            An expression to serialize.

        Returns
        -------
        dict
            Serialized expression.
        """
    if isinstance(expr, LiteralExpr):
        return self.serialize_literal(expr)
    elif isinstance(expr, CalciteInputRefExpr):
        return self.serialize_obj(expr)
    elif isinstance(expr, CalciteInputIdxExpr):
        return self.serialize_input_idx(expr)
    elif isinstance(expr, OpExpr):
        return self.serialize_typed_obj(expr)
    elif isinstance(expr, AggregateExpr):
        return self.serialize_typed_obj(expr)
    else:
        raise NotImplementedError('Can not serialize {}'.format(type(expr).__name__))