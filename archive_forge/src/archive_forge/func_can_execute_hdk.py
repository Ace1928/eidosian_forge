import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
@_inherit_docstrings(DFAlgNode.can_execute_hdk)
def can_execute_hdk(self) -> bool:
    if len(self.input) > 2:
        return False
    if len(self.input) == 0 or len(self.columns) == 0:
        return False
    dtypes = self.input[0]._dtypes.to_dict()
    if any((is_string_dtype(t) for t in dtypes.values())) or any((f._dtypes.to_dict() != dtypes for f in self.input[1:])):
        return False
    return True