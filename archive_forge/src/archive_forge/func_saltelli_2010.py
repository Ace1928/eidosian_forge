from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
def saltelli_2010(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saltelli2010 formulation.

    .. math::

        S_i = \\frac{1}{N} \\sum_{j=1}^N
        f(\\mathbf{B})_j (f(\\mathbf{AB}^{(i)})_j - f(\\mathbf{A})_j)

    .. math::

        S_{T_i} = \\frac{1}{N} \\sum_{j=1}^N
        (f(\\mathbf{A})_j - f(\\mathbf{AB}^{(i)})_j)^2

    Parameters
    ----------
    f_A, f_B : array_like (s, n)
        Function values at A and B, respectively
    f_AB : array_like (d, s, n)
        Function values at each of the AB pages

    Returns
    -------
    s, st : array_like (s, d)
        First order and total order Sobol' indices.

    References
    ----------
    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola. "Variance based sensitivity analysis of model
       output. Design and estimator for the total sensitivity index."
       Computer Physics Communications, 181(2):259-270,
       :doi:`10.1016/j.cpc.2009.09.018`, 2010.
    """
    var = np.var([f_A, f_B], axis=(0, -1))
    s = np.mean(f_B * (f_AB - f_A), axis=-1) / var
    st = 0.5 * np.mean((f_A - f_AB) ** 2, axis=-1) / var
    return (s.T, st.T)