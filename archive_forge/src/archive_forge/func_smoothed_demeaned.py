import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings
def smoothed_demeaned(self, exog):
    components = self.smoothed(exog)
    means = components.mean(0)
    constant = means.sum() + self.alpha
    components_demeaned = components - means
    return (components_demeaned, constant)