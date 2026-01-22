from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def explain_function_type(self, func):
    """
        Returns a string description of the type of a function
        """
    desc = []
    defns = []
    param = False
    if isinstance(func, types.Callable):
        sigs, param = func.get_call_signatures()
        defns.extend(sigs)
    elif func in self._functions:
        for tpl in self._functions[func]:
            param = param or hasattr(tpl, 'generic')
            defns.extend(getattr(tpl, 'cases', []))
    else:
        msg = 'No type info available for {func!r} as a callable.'
        desc.append(msg.format(func=func))
    if defns:
        desc = ['Known signatures:']
        for sig in defns:
            desc.append(' * {0}'.format(sig))
    return '\n'.join(desc)