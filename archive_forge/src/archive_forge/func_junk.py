from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
def junk():
    formula2 = 'deaths ~ C(agecat) + C(smokes) : C(agecat)'
    mod = Poisson.from_formula(formula2, data=data, exposure=data['pyears'].values)
    mod.fit()
    constraints = 'C(smokes)[T.1]:C(agecat)[3] = C(smokes)[T.1]:C(agec`at)[4]'
    import patsy
    lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constraints)
    R, q = (lc.coefs, lc.constants)
    mod.fit_constrained(R, q, fit_kwds={'method': 'bfgs'})
    formula1a = 'deaths ~ logpyears + smokes + C(agecat)'
    mod1a = Poisson.from_formula(formula1a, data=data)
    mod1a.fit()
    lc_1a = patsy.DesignInfo(mod1a.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
    mod1a.fit_constrained(lc_1a.coefs, lc_1a.constants, fit_kwds={'method': 'newton'})