import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def sentry_modified_builtin(self, inst, gvar):
    """
        Ensure that builtins are not modified.
        """
    if gvar.name == 'range' and gvar.value is not range:
        bad = True
    elif gvar.name == 'slice' and gvar.value is not slice:
        bad = True
    elif gvar.name == 'len' and gvar.value is not len:
        bad = True
    else:
        bad = False
    if bad:
        raise TypingError("Modified builtin '%s'" % gvar.name, loc=inst.loc)