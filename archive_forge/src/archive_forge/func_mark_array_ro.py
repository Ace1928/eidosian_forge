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
def mark_array_ro(tup):
    newtup = []
    for item in tup.types:
        if isinstance(item, types.Array):
            item = item.copy(readonly=True)
        elif isinstance(item, types.BaseAnonymousTuple):
            item = mark_array_ro(item)
        newtup.append(item)
    return types.BaseTuple.from_types(newtup)