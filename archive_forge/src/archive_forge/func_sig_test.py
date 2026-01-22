import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def sig_test(self, var_pos, nboot=50, nested_res=25, pivot=False):
    """
        Significance test for the variables in the regression.

        Parameters
        ----------
        var_pos : sequence
            The position of the variable in exog to be tested.

        Returns
        -------
        sig : str
            The level of significance:

                - `*` : at 90% confidence level
                - `**` : at 95% confidence level
                - `***` : at 99* confidence level
                - "Not Significant" : if not significant
        """
    var_pos = np.asarray(var_pos)
    ix_cont, ix_ord, ix_unord = _get_type_pos(self.var_type)
    if np.any(ix_cont[var_pos]):
        if np.any(ix_ord[var_pos]) or np.any(ix_unord[var_pos]):
            raise ValueError('Discrete variable in hypothesis. Must be continuous')
        Sig = TestRegCoefC(self, var_pos, nboot, nested_res, pivot)
    else:
        Sig = TestRegCoefD(self, var_pos, nboot)
    return Sig.sig