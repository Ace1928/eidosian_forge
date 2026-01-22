import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings
def linkinversepredict(self, exog):
    """expected value ? check new GLM, same as mu for given exog
        """
    return self.family.link.inverse(self.predict(exog))