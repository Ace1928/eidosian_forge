import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_conditional_mnlogit_grad():
    df = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula('y ~ 0 + x1 + x2', groups='g', data=df)
    for _ in range(5):
        za = np.random.normal(size=4)
        grad = model.score(za)
        ngrad = approx_fprime(za, model.loglike)
        assert_allclose(grad, ngrad, rtol=1e-05, atol=0.001)