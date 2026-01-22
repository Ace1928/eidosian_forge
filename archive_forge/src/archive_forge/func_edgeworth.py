from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def edgeworth(intervals):
    """
            Compute the Edgeworth expansion term of Sison & Glaz's formula
            (1) (approximated probability for multinomial proportions in a
            given box).
            """
    mu_r1, mu_r2, mu_r3, mu_r4 = (np.array([truncated_poisson_factorial_moment(interval, r, p) for interval, p in zip(intervals, counts)]) for r in range(1, 5))
    mu = mu_r1
    mu2 = mu_r2 + mu - mu ** 2
    mu3 = mu_r3 + mu_r2 * (3 - 3 * mu) + mu - 3 * mu ** 2 + 2 * mu ** 3
    mu4 = mu_r4 + mu_r3 * (6 - 4 * mu) + mu_r2 * (7 - 12 * mu + 6 * mu ** 2) + mu - 4 * mu ** 2 + 6 * mu ** 3 - 3 * mu ** 4
    g1 = mu3.sum() / mu2.sum() ** 1.5
    g2 = (mu4.sum() - 3 * (mu2 ** 2).sum()) / mu2.sum() ** 2
    x = (n - mu.sum()) / np.sqrt(mu2.sum())
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    H3 = x ** 3 - 3 * x
    H4 = x ** 4 - 6 * x ** 2 + 3
    H6 = x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15
    f = phi * (1 + g1 * H3 / 6 + g2 * H4 / 24 + g1 ** 2 * H6 / 72)
    return f / np.sqrt(mu2.sum())