import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def resolve_T(self, ary):
    if ary.ndim <= 1:
        retty = ary
    else:
        layout = {'C': 'F', 'F': 'C'}.get(ary.layout, 'A')
        retty = ary.copy(layout=layout)
    return retty