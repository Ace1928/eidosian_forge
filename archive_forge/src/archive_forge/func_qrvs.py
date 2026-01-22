import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def qrvs(self, size=None, d=None, qmc_engine=None):
    """
        Quasi-random variates of the given distribution.

        The `qmc_engine` is used to draw uniform quasi-random variates, and
        these are converted to quasi-random variates of the given distribution
        using inverse transform sampling.

        Parameters
        ----------
        size : int, tuple of ints, or None; optional
            Defines shape of random variates array. Default is ``None``.
        d : int or None, optional
            Defines dimension of uniform quasi-random variates to be
            transformed. Default is ``None``.
        qmc_engine : scipy.stats.qmc.QMCEngine(d=1), optional
            Defines the object to use for drawing
            quasi-random variates. Default is ``None``, which uses
            `scipy.stats.qmc.Halton(1)`.

        Returns
        -------
        rvs : ndarray or scalar
            Quasi-random variates. See Notes for shape information.

        Notes
        -----
        The shape of the output array depends on `size`, `d`, and `qmc_engine`.
        The intent is for the interface to be natural, but the detailed rules
        to achieve this are complicated.

        - If `qmc_engine` is ``None``, a `scipy.stats.qmc.Halton` instance is
          created with dimension `d`. If `d` is not provided, ``d=1``.
        - If `qmc_engine` is not ``None`` and `d` is ``None``, `d` is
          determined from the dimension of the `qmc_engine`.
        - If `qmc_engine` is not ``None`` and `d` is not ``None`` but the
          dimensions are inconsistent, a ``ValueError`` is raised.
        - After `d` is determined according to the rules above, the output
          shape is ``tuple_shape + d_shape``, where:

              - ``tuple_shape = tuple()`` if `size` is ``None``,
              - ``tuple_shape = (size,)`` if `size` is an ``int``,
              - ``tuple_shape = size`` if `size` is a sequence,
              - ``d_shape = tuple()`` if `d` is ``None`` or `d` is 1, and
              - ``d_shape = (d,)`` if `d` is greater than 1.

        The elements of the returned array are part of a low-discrepancy
        sequence. If `d` is 1, this means that none of the samples are truly
        independent. If `d` > 1, each slice ``rvs[..., i]`` will be of a
        quasi-independent sequence; see `scipy.stats.qmc.QMCEngine` for
        details. Note that when `d` > 1, the samples returned are still those
        of the provided univariate distribution, not a multivariate
        generalization of that distribution.

        """
    qmc_engine, d = _validate_qmc_input(qmc_engine, d, self.random_state)
    try:
        if size is None:
            tuple_size = (1,)
        else:
            tuple_size = tuple(size)
    except TypeError:
        tuple_size = (size,)
    N = 1 if size is None else np.prod(size)
    u = qmc_engine.random(N)
    if self._mirror_uniform:
        u = 1 - u
    qrvs = self._ppf(u)
    if self._rvs_transform is not None:
        qrvs = self._rvs_transform(qrvs, *self._frozendist.args)
    if size is None:
        qrvs = qrvs.squeeze()[()]
    elif d == 1:
        qrvs = qrvs.reshape(tuple_size)
    else:
        qrvs = qrvs.reshape(tuple_size + (d,))
    return self.loc + self.scale * qrvs