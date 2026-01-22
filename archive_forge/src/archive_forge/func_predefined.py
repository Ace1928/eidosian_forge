from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def predefined(self, y0, xout, params=(), **kwargs):
    """ Integrate with user chosen output.

        Parameters
        ----------
        integrator : str
            See :meth:`integrate`.
        y0 : array_like
            See :meth:`integrate`.
        xout : array_like
        params : array_like
            See :meth:`integrate`.
        \\*\\*kwargs:
            See :meth:`integrate`

        Returns
        -------
        Length 2 tuple : (yout, info)
            See :meth:`integrate`.
        """
    xout, yout, info = self.integrate(xout, y0, params=params, force_predefined=True, **kwargs)
    return (yout, info)