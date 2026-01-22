import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
def lr_test(self):
    """
        generic likelihood ratio test between nested models

            \\begin{align}
            D & = -2(\\ln(\\text{likelihood for null model}) - \\ln(\\text{likelihood for alternative model})) \\\\
            & = -2\\ln\\left( \\frac{\\text{likelihood for null model}}{\\text{likelihood for alternative model}} \\right).
            \\end{align}

        is distributed as chisquare with df equal to difference in number of parameters or equivalently
        difference in residual degrees of freedom  (sign?)

        TODO: put into separate function
        """
    if not hasattr(self, 'lsjoint'):
        self.fitjoint()
    if not hasattr(self, 'lspooled'):
        self.fitpooled()
    loglikejoint = self.lsjoint.llf
    loglikepooled = self.lspooled.llf
    lrstat = -2 * (loglikepooled - loglikejoint)
    lrdf = self.lspooled.df_resid - self.lsjoint.df_resid
    lrpval = stats.chi2.sf(lrstat, lrdf)
    return (lrstat, lrpval, lrdf)