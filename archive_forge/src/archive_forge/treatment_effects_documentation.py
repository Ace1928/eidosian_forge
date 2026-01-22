import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent

        ATE and POM from inverse probability weighted regression adjustment.

        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        