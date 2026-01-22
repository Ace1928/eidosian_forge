import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def svar_ma_rep(self, maxn=10, P=None):
    """

        Compute Structural MA coefficient matrices using MLE
        of A, B
        """
    if P is None:
        A_solve = self.A_solve
        B_solve = self.B_solve
        P = np.dot(npl.inv(A_solve), B_solve)
    ma_mats = self.ma_rep(maxn=maxn)
    return np.array([np.dot(coefs, P) for coefs in ma_mats])