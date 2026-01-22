import sys
from _pydevd_bundle.pydevd_constants import PANDAS_MAX_ROWS, PANDAS_MAX_COLS, PANDAS_MAX_COLWIDTH
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_resolver import inspect, MethodWrapperType
from _pydevd_bundle.pydevd_utils import Timer
from .pydevd_helpers import find_mod_attr
from contextlib import contextmanager
@contextmanager
def customize_pandas_options():
    custom_options = []
    from pandas import get_option
    max_rows = get_option('display.max_rows')
    max_cols = get_option('display.max_columns')
    max_colwidth = get_option('display.max_colwidth')
    if max_rows is None or max_rows > PANDAS_MAX_ROWS:
        custom_options.append('display.max_rows')
        custom_options.append(PANDAS_MAX_ROWS)
    if max_cols is None or max_cols > PANDAS_MAX_COLS:
        custom_options.append('display.max_columns')
        custom_options.append(PANDAS_MAX_COLS)
    if max_colwidth is None or max_colwidth > PANDAS_MAX_COLWIDTH:
        custom_options.append('display.max_colwidth')
        custom_options.append(PANDAS_MAX_COLWIDTH)
    if custom_options:
        from pandas import option_context
        with option_context(*custom_options):
            yield
    else:
        yield