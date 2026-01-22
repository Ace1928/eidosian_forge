from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def input_ids(self):
    """
            Get ids of all input nodes.

            Returns
            -------
            list of int
            """
    return [x.id for x in self.input_nodes]