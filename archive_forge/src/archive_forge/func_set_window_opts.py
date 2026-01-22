import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
def set_window_opts(self, partition_keys, order_keys, order_ascending, na_pos):
    """
        Set the window function options.

        Parameters
        ----------
        partition_keys : list of BaseExpr
        order_keys : list of BaseExpr
        order_ascending : list of bool
        na_pos : {"FIRST", "LAST"}
        """
    self.is_rows = True
    self.partition_keys = partition_keys
    self.order_keys = []
    for key, asc in zip(order_keys, order_ascending):
        key = {'field': key, 'direction': 'ASCENDING' if asc else 'DESCENDING', 'nulls': na_pos}
        self.order_keys.append(key)
    self.lower_bound = {'unbounded': True, 'preceding': True, 'following': False, 'is_current_row': False, 'offset': None, 'order_key': 0}
    self.upper_bound = {'unbounded': False, 'preceding': False, 'following': False, 'is_current_row': True, 'offset': None, 'order_key': 1}