from __future__ import annotations
import numpy as np
def round_to_sigfigs(num, sig_figs):
    """Rounds a number rounded to a specific number of significant
    figures instead of to a specific precision.
    """
    if not isinstance(sig_figs, int):
        raise TypeError('Number of significant figures must be integer')
    if sig_figs < 1:
        raise ValueError('Number of significant figures must be positive')
    if num == 0:
        return num
    prec = int(sig_figs - np.ceil(np.log10(np.absolute(num))))
    return round(num, prec)