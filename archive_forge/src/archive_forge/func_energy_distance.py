import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def energy_distance(u_values, v_values, u_weights=None, v_weights=None):
    """Compute the energy distance between two 1D distributions.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The energy distance between two distributions :math:`u` and :math:`v`, whose
    respective CDFs are :math:`U` and :math:`V`, equals to:

    .. math::

        D(u, v) = \\left( 2\\mathbb E|X - Y| - \\mathbb E|X - X'| -
        \\mathbb E|Y - Y'| \\right)^{1/2}

    where :math:`X` and :math:`X'` (resp. :math:`Y` and :math:`Y'`) are
    independent random variables whose probability distribution is :math:`u`
    (resp. :math:`v`).

    Sometimes the square of this quantity is referred to as the "energy
    distance" (e.g. in [2]_, [4]_), but as noted in [1]_ and [3]_, only the
    definition above satisfies the axioms of a distance function (metric).

    As shown in [2]_, for one-dimensional real-valued variables, the energy
    distance is linked to the non-distribution-free version of the Cramér-von
    Mises distance:

    .. math::

        D(u, v) = \\sqrt{2} l_2(u, v) = \\left( 2 \\int_{-\\infty}^{+\\infty} (U-V)^2
        \\right)^{1/2}

    Note that the common Cramér-von Mises criterion uses the distribution-free
    version of the distance. See [2]_ (section 2), for more details about both
    versions of the distance.

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Rizzo, Szekely "Energy distance." Wiley Interdisciplinary Reviews:
           Computational Statistics, 8(1):27-38 (2015).
    .. [2] Szekely "E-statistics: The energy of statistical samples." Bowling
           Green State University, Department of Mathematics and Statistics,
           Technical Report 02-16 (2002).
    .. [3] "Energy distance", https://en.wikipedia.org/wiki/Energy_distance
    .. [4] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    Examples
    --------
    >>> from scipy.stats import energy_distance
    >>> energy_distance([0], [2])
    2.0000000000000004
    >>> energy_distance([0, 8], [0, 8], [3, 1], [2, 2])
    1.0000000000000002
    >>> energy_distance([0.7, 7.4, 2.4, 6.8], [1.4, 8. ],
    ...                 [2.1, 4.2, 7.4, 8. ], [7.6, 8.8])
    0.88003340976158217

    """
    return np.sqrt(2) * _cdf_distance(2, u_values, v_values, u_weights, v_weights)