from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
create the spline basis for new observations

        The main use of this stateful transformation is for prediction
        using the same specification of the spline basis.

        Parameters
        ----------
        x_new: ndarray
            observations of the underlying explanatory variable

        Returns
        -------
        basis : ndarray
            design matrix for the spline basis for given ``x_new``.
        