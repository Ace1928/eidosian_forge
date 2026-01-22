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
def bind_wrappers(cls):
    """
    Wrap class methods.

    Decorator allows to fallback to the parent query compiler methods when unsupported
    data types are used in a frame.

    Returns
    -------
    class
    """
    exclude = set(['__init__', 'to_pandas', 'from_pandas', 'from_arrow', 'default_to_pandas', '_get_index', '_set_index', '_get_columns', '_set_columns'])
    for name, method in cls.__dict__.items():
        if name in exclude:
            continue
        if callable(method):
            setattr(cls, name, build_method_wrapper(name, method))
    return cls