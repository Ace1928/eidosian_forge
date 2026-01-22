import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def sentry_shape_scalar(ty):
    if ty in types.number_domain:
        if not isinstance(ty, types.Integer):
            raise TypeError('reshape() arg cannot be {0}'.format(ty))
        return True
    else:
        return False