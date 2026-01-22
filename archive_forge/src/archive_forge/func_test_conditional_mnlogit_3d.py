import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_conditional_mnlogit_3d():
    df = gen_mnlogit(90)
    df['x3'] = np.random.normal(size=df.shape[0])
    model = ConditionalMNLogit.from_formula('y ~ 0 + x1 + x2 + x3', groups='g', data=df)
    result = model.fit()
    assert_allclose(result.params, np.asarray([[0.729629, -1.633673], [1.879019, -1.327163], [-0.114124, -0.109378]]), atol=1e-05, rtol=1e-05)
    assert_allclose(result.bse, np.asarray([[0.682965, 0.60472], [0.672947, 0.42401], [0.722631, 0.33663]]), atol=1e-05, rtol=1e-05)
    result.summary()