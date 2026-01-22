from statsmodels.compat.python import lzip
import numpy as np
from statsmodels.tools.testing import Holder
def pval_corrected(self, method=None):
    """p-values corrected for multiple testing problem

        This uses the default p-value correction of the instance stored in
        ``self.multitest_method`` if method is None.

        """
    import statsmodels.stats.multitest as smt
    if method is None:
        method = self.multitest_method
    return smt.multipletests(self.pvals_raw, method=method)[1]