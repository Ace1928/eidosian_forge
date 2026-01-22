from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def ref_idx(self, frame, col):
    """
            Translate input column into ``CalciteInputIdxExpr``.

            Parameters
            ----------
            frame : DFAlgNode
                An input frame.
            col : str
                An input column.

            Returns
            -------
            CalciteInputIdxExpr
            """
    return CalciteInputIdxExpr(self._idx(frame, col))