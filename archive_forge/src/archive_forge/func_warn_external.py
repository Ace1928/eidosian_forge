import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def warn_external(message, category=None):
    """
    `warnings.warn` wrapper that sets *stacklevel* to "outside Matplotlib".

    The original emitter of the warning can be obtained by patching this
    function back to `warnings.warn`, i.e. ``_api.warn_external =
    warnings.warn`` (or ``functools.partial(warnings.warn, stacklevel=2)``,
    etc.).
    """
    frame = sys._getframe()
    for stacklevel in itertools.count(1):
        if frame is None:
            break
        if not re.match('\\A(matplotlib|mpl_toolkits)(\\Z|\\.(?!tests\\.))', frame.f_globals.get('__name__', '')):
            break
        frame = frame.f_back
    del frame
    warnings.warn(message, category, stacklevel)