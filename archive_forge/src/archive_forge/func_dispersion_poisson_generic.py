import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def dispersion_poisson_generic(results, exog_new_test, exog_new_control=None, include_score=False, use_endog=True, cov_type='HC3', cov_kwds=None, use_t=False):
    """A variable addition test for the variance function

    .. deprecated:: 0.14

       dispersion_poisson_generic moved to discrete._diagnostic_count

    This uses an artificial regression to calculate a variant of an LM or
    generalized score test for the specification of the variance assumption
    in a Poisson model. The performed test is a Wald test on the coefficients
    of the `exog_new_test`.

    Warning: insufficiently tested, especially for options
    """
    msg = 'dispersion_poisson_generic here is deprecated, use the version in discrete._diagnostic_count'
    warnings.warn(msg, FutureWarning)
    from statsmodels.discrete._diagnostics_count import _test_poisson_dispersion_generic
    res_test = _test_poisson_dispersion_generic(results, exog_new_test, exog_new_control=exog_new_control, include_score=include_score, use_endog=use_endog, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    return res_test