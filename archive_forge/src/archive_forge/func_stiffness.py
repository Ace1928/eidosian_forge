from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def stiffness(self, xyp=None, eigenvals_cb=None):
    """ Running stiffness ratio from last integration.

        Calculate sittness ratio, i.e. the ratio between the largest and
        smallest absolute eigenvalue of the jacobian matrix. The user may
        supply their own routine for calculating the eigenvalues, or they
        will be calculated from the SVD (singular value decomposition).
        Note that calculating the SVD for any but the smallest Jacobians may
        prove to be prohibitively expensive.

        Parameters
        ----------
        xyp : length 3 tuple (default: None)
            internal_xout, internal_yout, internal_params, taken
            from last integration if not specified.
        eigenvals_cb : callback (optional)
            Signature (x, y, p) (internal variables), when not provided an
            internal routine will use ``self.j_cb`` and ``scipy.linalg.svd``.

        """
    if eigenvals_cb is None:
        if self.odesys.band is not None:
            raise NotImplementedError
        eigenvals_cb = self.odesys._jac_eigenvals_svd
    if xyp is None:
        x, y, intern_p = self._internals()
    else:
        x, y, intern_p = self.pre_process(*xyp)
    singular_values = []
    for xval, yvals in zip(x, y):
        singular_values.append(eigenvals_cb(xval, yvals, intern_p))
    return np.abs(singular_values).max(axis=-1) / np.abs(singular_values).min(axis=-1)