import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_obj(self, obj):
    """
        Serialize an object into a dictionary.

        Add all non-hidden attributes (not starting with '_') of the object
        to the output dictionary.

        Parameters
        ----------
        obj : object
            An object to serialize.

        Returns
        -------
        dict
            Serialized object.
        """
    res = {}
    for k, v in obj.__dict__.items():
        if k[0] != '_':
            if k == 'op' and isinstance(obj, OpExpr) and (v == '//'):
                res[k] = '/'
            else:
                res[k] = self.serialize_item(v)
    return res