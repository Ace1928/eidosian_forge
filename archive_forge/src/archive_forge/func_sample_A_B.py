from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
def sample_A_B(n: IntNumber, dists: list[PPFDist], random_state: SeedType=None) -> np.ndarray:
    """Sample two matrices A and B.

    Uses a Sobol' sequence with 2`d` columns to have 2 uncorrelated matrices.
    This is more efficient than using 2 random draw of Sobol'.
    See sec. 5 from [1]_.

    Output shape is (d, n).

    References
    ----------
    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola. "Variance based sensitivity analysis of model
       output. Design and estimator for the total sensitivity index."
       Computer Physics Communications, 181(2):259-270,
       :doi:`10.1016/j.cpc.2009.09.018`, 2010.
    """
    d = len(dists)
    A_B = qmc.Sobol(d=2 * d, seed=random_state, bits=64).random(n).T
    A_B = A_B.reshape(2, d, -1)
    try:
        for d_, dist in enumerate(dists):
            A_B[:, d_] = dist.ppf(A_B[:, d_])
    except AttributeError as exc:
        message = 'Each distribution in `dists` must have method `ppf`.'
        raise ValueError(message) from exc
    return A_B