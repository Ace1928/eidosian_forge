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
def translate_input(self, mapper):
    """
        Make a deep copy of the expression translating input nodes using `mapper`.

        The default implementation builds a copy and recursively run
        translation for all its operands. For leaf expressions
        `_translate_input` is called.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input columns translation.

        Returns
        -------
        BaseExpr
            The expression copy with translated input columns.
        """
    res = None
    gen = self.nested_expressions()
    for expr in gen:
        res = gen.send(expr.translate_input(mapper))
    return self._translate_input(mapper) if res is None else res