from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
@cache_readonly
def cov_params_default(self):
    beta = self.beta
    if self.det_coef_coint.size > 0:
        beta = vstack((beta, self.det_coef_coint))
    dt = self.deterministic
    num_det = ('co' in dt) + ('lo' in dt)
    num_det += self.seasons - 1 if self.seasons else 0
    if self.exog is not None:
        num_det += self.exog.shape[1]
    b_id = scipy.linalg.block_diag(beta, np.identity(self.neqs * (self.k_ar - 1) + num_det))
    y_lag1 = self._y_lag1
    b_y = beta.T.dot(y_lag1)
    omega11 = b_y.dot(b_y.T)
    omega12 = b_y.dot(self._delta_x.T)
    omega21 = omega12.T
    omega22 = self._delta_x.dot(self._delta_x.T)
    omega = np.bmat([[omega11, omega12], [omega21, omega22]]).A
    mat1 = b_id.dot(inv(omega)).dot(b_id.T)
    return np.kron(mat1, self.sigma_u)