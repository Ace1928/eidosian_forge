from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
def npconvert(ctx, x):
    """
        Converts *x* to an ``mpf`` or ``mpc``. *x* should be a numpy
        scalar.
        """
    import numpy as np
    if isinstance(x, np.integer):
        return ctx.make_mpf(from_int(int(x)))
    if isinstance(x, np.floating):
        return ctx.make_mpf(from_npfloat(x))
    if isinstance(x, np.complexfloating):
        return ctx.make_mpc((from_npfloat(x.real), from_npfloat(x.imag)))
    raise TypeError('cannot create mpf from ' + repr(x))