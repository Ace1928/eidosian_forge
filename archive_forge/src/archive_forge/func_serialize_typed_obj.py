import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_typed_obj(self, obj):
    """
        Serialize an object and its dtype into a dictionary.

        Similar to `serialize_obj` but also include '_dtype' field
        of the object under 'type' key.

        Parameters
        ----------
        obj : object
            An object to serialize.

        Returns
        -------
        dict
            Serialized object.
        """
    res = self.serialize_obj(obj)
    res['type'] = self.serialize_dtype(obj._dtype)
    return res