import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest
@pytest.mark.slow
def test_framing_example_formula():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', 'framing.csv'))
    outcome_model = sm.GLM.from_formula('cong_mesg ~ emo + treat + age + educ + gender + income', data, family=sm.families.Binomial(link=sm.families.links.Probit()))
    mediator_model = sm.OLS.from_formula('emo ~ treat + age + educ + gender + income', data)
    med = Mediation(outcome_model, mediator_model, 'treat', 'emo', outcome_fit_kwargs={'atol': 1e-11})
    np.random.seed(4231)
    med_rslt = med.fit(method='boot', n_rep=100)
    diff = np.asarray(med_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-06)
    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=100)
    diff = np.asarray(med_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-06)