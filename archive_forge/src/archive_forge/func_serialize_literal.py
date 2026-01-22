import json
import numpy as np
from pandas.core.dtypes.common import is_datetime64_dtype
from modin.error_message import ErrorMessage
from .calcite_algebra import (
from .expr import AggregateExpr, BaseExpr, LiteralExpr, OpExpr
def serialize_literal(self, literal):
    """
        Serialize ``LiteralExpr`` into a dictionary.

        Parameters
        ----------
        literal : LiteralExpr
            A literal to serialize.

        Returns
        -------
        dict
            Serialized literal.
        """
    val = literal.val
    if val is None:
        return {'literal': None, 'type': 'BIGINT', 'target_type': 'BIGINT', 'scale': 0, 'precision': 19, 'type_scale': 0, 'type_precision': 19}
    if type(val) is str:
        return {'literal': val, 'type': 'CHAR', 'target_type': 'CHAR', 'scale': -2147483648, 'precision': len(val), 'type_scale': -2147483648, 'type_precision': len(val)}
    if type(val) in self._INT_OPTS.keys():
        target_type, precision = self.opts_for_int_type(type(val))
        return {'literal': int(val), 'type': 'DECIMAL', 'target_type': target_type, 'scale': 0, 'precision': len(str(val)), 'type_scale': 0, 'type_precision': precision}
    if type(val) in (float, np.float64):
        if np.isnan(val):
            return {'literal': None, 'type': 'DOUBLE', 'target_type': 'DOUBLE', 'scale': 0, 'precision': 19, 'type_scale': 0, 'type_precision': 19}
        str_val = f'{val:f}'
        precision = len(str_val) - 1
        scale = precision - str_val.index('.')
        return {'literal': int(str_val.replace('.', '')), 'type': 'DECIMAL', 'target_type': 'DOUBLE', 'scale': scale, 'precision': precision, 'type_scale': -2147483648, 'type_precision': 15}
    if type(val) is bool:
        return {'literal': val, 'type': 'BOOLEAN', 'target_type': 'BOOLEAN', 'scale': -2147483648, 'precision': 1, 'type_scale': -2147483648, 'type_precision': 1}
    if isinstance(val, np.datetime64):
        unit = np.datetime_data(val)[0]
        precision = self._TIMESTAMP_PRECISION.get(unit, None)
        if precision is not None:
            return {'literal': int(val.astype(np.int64)), 'type': 'TIMESTAMP', 'target_type': 'TIMESTAMP', 'scale': -2147483648, 'precision': precision, 'type_scale': -2147483648, 'type_precision': precision}
    raise NotImplementedError(f'Can not serialize {type(val).__name__}')