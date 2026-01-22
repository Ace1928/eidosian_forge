from functools import wraps
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_bool_dtype, is_integer_dtype
from modin.core.storage_formats import BaseQueryCompiler
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
def groupby_size(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
    if len(self.columns) == 0:
        raise NotImplementedError('Grouping on empty frame or on index level is not yet implemented.')
    groupby_kwargs = groupby_kwargs.copy()
    as_index = groupby_kwargs.get('as_index', True)
    groupby_kwargs['as_index'] = True
    new_frame = self._modin_frame.groupby_agg(by, axis, {self._modin_frame.columns[0]: 'size'}, groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)
    if as_index:
        shape_hint = 'column'
        new_frame = new_frame._set_columns([MODIN_UNNAMED_SERIES_LABEL])
    else:
        shape_hint = None
        new_frame = new_frame._set_columns(['size']).reset_index(drop=False)
    return self.__constructor__(new_frame, shape_hint=shape_hint)