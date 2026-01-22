from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def loading_constraints(self, i):
    """
        Matrix formulation of quarterly variables' factor loading constraints.

        Parameters
        ----------
        i : int
            Index of the `endog` variable to compute constraints for.

        Returns
        -------
        R : array (k_constraints, k_factors * 5)
        q : array (k_constraints,)

        Notes
        -----
        If the factors were known, then the factor loadings for the ith
        quarterly variable would be computed by a linear regression of the form

        y_i = A_i' f + B_i' L1.f + C_i' L2.f + D_i' L3.f + E_i' L4.f

        where:

        - f is (k_i x 1) and collects all of the factors that load on y_i
        - L{j}.f is (k_i x 1) and collects the jth lag of each factor
        - A_i, ..., E_i are (k_i x 1) and collect factor loadings

        As the observed variable is quarterly while the factors are monthly, we
        want to restrict the estimated regression coefficients to be:

        y_i = A_i f + 2 A_i L1.f + 3 A_i L2.f + 2 A_i L3.f + A_i L4.f

        Stack the unconstrained coefficients: \\Lambda_i = [A_i' B_i' ... E_i']'

        Then the constraints can be written as follows, for l = 1, ..., k_i

        - 2 A_{i,l} - B_{i,l} = 0
        - 3 A_{i,l} - C_{i,l} = 0
        - 2 A_{i,l} - D_{i,l} = 0
        - A_{i,l} - E_{i,l} = 0

        So that k_constraints = 4 * k_i. In matrix form the constraints are:

        .. math::

            R \\Lambda_i = q

        where :math:`\\Lambda_i` is shaped `(k_i * 5,)`, :math:`R` is shaped
        `(k_constraints, k_i * 5)`, and :math:`q` is shaped `(k_constraints,)`.


        For example, for the case that k_i = 2, we can write:

        |  2 0   -1  0    0  0    0  0    0  0  |   | A_{i,1} |     | 0 |
        |  0 2    0 -1    0  0    0  0    0  0  |   | A_{i,2} |     | 0 |
        |  3 0    0  0   -1  0    0  0    0  0  |   | B_{i,1} |     | 0 |
        |  0 3    0  0    0 -1    0  0    0  0  |   | B_{i,2} |     | 0 |
        |  2 0    0  0    0  0   -1  0    0  0  |   | C_{i,1} |  =  | 0 |
        |  0 2    0  0    0  0    0 -1    0  0  |   | C_{i,2} |     | 0 |
        |  1 0    0  0    0  0    0  0   -1  0  |   | D_{i,1} |     | 0 |
        |  0 1    0  0    0  0    0  0    0 -1  |   | D_{i,2} |     | 0 |
                                                    | E_{i,1} |     | 0 |
                                                    | E_{i,2} |     | 0 |

        """
    if i < self.k_endog_M:
        raise ValueError('No constraints for monthly variables.')
    if i not in self._loading_constraints:
        k_factors = self.endog_factor_map.iloc[i].sum()
        R = np.zeros((k_factors * 4, k_factors * 5))
        q = np.zeros(R.shape[0])
        multipliers = np.array([1, 2, 3, 2, 1])
        R[:, :k_factors] = np.reshape((multipliers[1:] * np.eye(k_factors)[..., None]).T, (k_factors * 4, k_factors))
        R[:, k_factors:] = np.diag([-1] * (k_factors * 4))
        self._loading_constraints[i] = (R, q)
    return self._loading_constraints[i]