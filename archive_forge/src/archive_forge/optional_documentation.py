import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,

    The handling of optional->optional cast must be special cased for
    correct propagation of None value.  Given type T and U. casting of
    T? to U? (? denotes optional) should always succeed.   If the from-value
    is None, the None value the casted value (U?) should be None; otherwise,
    the from-value is casted to U. This is different from casting T? to U,
    which requires the from-value must not be None.
    