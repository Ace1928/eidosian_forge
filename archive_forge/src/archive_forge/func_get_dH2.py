import numpy as np
from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi
def get_dH2(lab1, lab2, *, channel_axis=-1):
    """squared hue difference term occurring in deltaE_cmc and deltaE_ciede94

    Despite its name, "dH" is not a simple difference of hue values.  We avoid
    working directly with the hue value, since differencing angles is
    troublesome.  The hue term is usually written as:
        c1 = sqrt(a1**2 + b1**2)
        c2 = sqrt(a2**2 + b2**2)
        term = (a1-a2)**2 + (b1-b2)**2 - (c1-c2)**2
        dH = sqrt(term)

    However, this has poor roundoff properties when a or b is dominant.
    Instead, ab is a vector with elements a and b.  The same dH term can be
    re-written as:
        |ab1-ab2|**2 - (|ab1| - |ab2|)**2
    and then simplified to:
        2*|ab1|*|ab2| - 2*dot(ab1, ab2)
    """
    input_is_float_32 = _supported_float_type((lab1.dtype, lab2.dtype)) == np.float32
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=False)
    a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[1:3]
    a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[1:3]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    term = C1 * C2 - (a1 * a2 + b1 * b2)
    out = 2 * term
    if input_is_float_32:
        out = out.astype(np.float32)
    return out