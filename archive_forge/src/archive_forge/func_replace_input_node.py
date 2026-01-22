from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def replace_input_node(self, frame, node, new_cols):
    """
            Use `node` as an input node for references to columns of `frame`.

            Parameters
            ----------
            frame : DFAlgNode
                Replaced input frame.
            node : CalciteBaseNode
                A new node to use.
            new_cols : list of str
                A new columns list to use.
            """
    self.replacements[frame] = new_cols