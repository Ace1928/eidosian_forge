from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
def f_ishigami(x: npt.ArrayLike) -> np.ndarray:
    """Ishigami function.

    .. math::

        Y(\\mathbf{x}) = \\sin x_1 + 7 \\sin^2 x_2 + 0.1 x_3^4 \\sin x_1

    with :math:`\\mathbf{x} \\in [-\\pi, \\pi]^3`.

    Parameters
    ----------
    x : array_like ([x1, x2, x3], n)

    Returns
    -------
    f : array_like (n,)
        Function evaluation.

    References
    ----------
    .. [1] Ishigami, T. and T. Homma. "An importance quantification technique
       in uncertainty analysis for computer models." IEEE,
       :doi:`10.1109/ISUMA.1990.151285`, 1990.
    """
    x = np.atleast_2d(x)
    f_eval = np.sin(x[0]) + 7 * np.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * np.sin(x[0])
    return f_eval