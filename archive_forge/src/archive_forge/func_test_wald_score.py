import numpy as np
from numpy.testing import assert_allclose
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.stats._diagnostic_other as diao
import statsmodels.discrete._diagnostics_count as diac
from statsmodels.base._parameter_inference import score_test
def test_wald_score(self):
    super().test_wald_score()