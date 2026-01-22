import sys
from _pydevd_bundle.pydevd_constants import PANDAS_MAX_ROWS, PANDAS_MAX_COLS, PANDAS_MAX_COLWIDTH
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_resolver import inspect, MethodWrapperType
from _pydevd_bundle.pydevd_utils import Timer
from .pydevd_helpers import find_mod_attr
from contextlib import contextmanager

        :param context:
            This is the context in which the variable is being requested. Valid values:
                "watch",
                "repl",
                "hover",
                "clipboard"
        