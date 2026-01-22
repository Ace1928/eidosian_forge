from __future__ import (absolute_import, division, print_function)
from math import log
import numpy as np
def info_vlines(ax, xout, info, vline_colors=('maroon', 'purple'), vline_keys=('steps', 'rhs_xvals', 'jac_xvals'), post_proc=None, alpha=None, fpes=None, every=None):
    """ Plot vertical lines in the background

    Parameters
    ----------
    ax : axes
    xout : array_like
    info : dict
    vline_colors : iterable of str
    vline_keys : iterable of str
        Choose from ``'steps', 'rhs_xvals', 'jac_xvals',
        'fe_underflow', 'fe_overflow', 'fe_invalid', 'fe_divbyzero'``.
    vline_post_proc : callable
    alpha : float

    """
    nvk = len(vline_keys)
    for idx, key in enumerate(vline_keys):
        if key == 'steps':
            vlines = xout
        elif key.startswith('fe_'):
            if fpes is None:
                raise ValueError('Need fpes when vline_keys contain fe_*')
            vlines = xout[info['fpes'] & fpes[key.upper()] > 0]
        else:
            vlines = info[key] if post_proc is None else post_proc(info[key])
        if alpha is None:
            alpha = 0.01 + 1 / log(len(vlines) + 3)
        if every is None:
            ln_np1 = log(len(vlines) + 1)
            every = min(round((ln_np1 - 4) / log(2)), 1)
        ax.vlines(vlines[::every], idx / nvk + 0.002, (idx + 1) / nvk - 0.002, colors=vline_colors[idx % len(vline_colors)], alpha=alpha, transform=ax.get_xaxis_transform())
    right_hand_ylabels(ax, [k[3] if k.startswith('fe_') else k[0] for k in vline_keys])