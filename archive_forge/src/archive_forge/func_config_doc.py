import ctypes
import json
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, cast
from ._typing import _F
from .core import _LIB, _check_call, c_str, py_str
def config_doc(*, header: Optional[str]=None, extra_note: Optional[str]=None, parameters: Optional[str]=None, returns: Optional[str]=None, see_also: Optional[str]=None) -> Callable[[_F], _F]:
    """Decorator to format docstring for config functions.

    Parameters
    ----------
    header: str
        An introducion to the function
    extra_note: str
        Additional notes
    parameters: str
        Parameters of the function
    returns: str
        Return value
    see_also: str
        Related functions
    """
    doc_template = '\n    {header}\n\n    Global configuration consists of a collection of parameters that can be applied in the\n    global scope. See :ref:`global_config` for the full list of parameters supported in\n    the global configuration.\n\n    {extra_note}\n\n    .. versionadded:: 1.4.0\n    '
    common_example = '\n    Example\n    -------\n\n    .. code-block:: python\n\n        import xgboost as xgb\n\n        # Show all messages, including ones pertaining to debugging\n        xgb.set_config(verbosity=2)\n\n        # Get current value of global configuration\n        # This is a dict containing all parameters in the global configuration,\n        # including \'verbosity\'\n        config = xgb.get_config()\n        assert config[\'verbosity\'] == 2\n\n        # Example of using the context manager xgb.config_context().\n        # The context manager will restore the previous value of the global\n        # configuration upon exiting.\n        with xgb.config_context(verbosity=0):\n            # Suppress warning caused by model generated with XGBoost version < 1.0.0\n            bst = xgb.Booster(model_file=\'./old_model.bin\')\n        assert xgb.get_config()[\'verbosity\'] == 2  # old value restored\n\n    Nested configuration context is also supported:\n\n    Example\n    -------\n\n    .. code-block:: python\n\n        with xgb.config_context(verbosity=3):\n            assert xgb.get_config()["verbosity"] == 3\n            with xgb.config_context(verbosity=2):\n                assert xgb.get_config()["verbosity"] == 2\n\n        xgb.set_config(verbosity=2)\n        assert xgb.get_config()["verbosity"] == 2\n        with xgb.config_context(verbosity=3):\n            assert xgb.get_config()["verbosity"] == 3\n    '

    def none_to_str(value: Optional[str]) -> str:
        return '' if value is None else value

    def config_doc_decorator(func: _F) -> _F:
        func.__doc__ = doc_template.format(header=none_to_str(header), extra_note=none_to_str(extra_note)) + none_to_str(parameters) + none_to_str(returns) + none_to_str(common_example) + none_to_str(see_also)

        @wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return cast(_F, wrap)
    return config_doc_decorator