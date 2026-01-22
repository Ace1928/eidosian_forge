import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def opts_for_int_type(self, int_type):
    """
        Get serialization params for an integer type.

        Return a SQL type name and a number of meaningful decimal digits
        for an integer type.

        Parameters
        ----------
        int_type : type
            An integer type to describe.

        Returns
        -------
        tuple
        """
    try:
        _warn_if_unsigned(int_type)
        return self._INT_OPTS[int_type]
    except KeyError:
        raise NotImplementedError(f'Unsupported integer type {int_type.__name__}')