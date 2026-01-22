import importlib
import logging
import sys
import textwrap
from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from packaging.version import Version
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import DeveloperAPI
from ray.widgets import Template
@DeveloperAPI
def repr_with_fallback(*notebook_deps: Iterable[Union[str, Optional[str]]]) -> Callable[[F], F]:
    """Decorator which strips rich notebook output from mimebundles in certain cases.

    Fallback to plaintext and don't use rich output in the following cases:
    1. In a notebook environment and the appropriate dependencies are not installed.
    2. In a ipython shell environment.
    3. In Google Colab environment.
        See https://github.com/googlecolab/colabtools/ issues/60 for more information
        about the status of this issue.

    Args:
        notebook_deps: The required dependencies and version for notebook environment.

    Returns:
        A function that returns the usual _repr_mimebundle_, unless any of the 3
        conditions above hold, in which case it returns a mimebundle that only contains
        a single text/plain mimetype.
    """
    message = 'Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.'
    if _can_display_ipywidgets(*notebook_deps, message=message):

        def wrapper(func: F) -> F:

            @wraps(func)
            def wrapped(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            return wrapped
    else:

        def wrapper(func: F) -> F:

            @wraps(func)
            def wrapped(self, *args, **kwargs):
                return {'text/plain': repr(self)}
            return wrapped
    return wrapper