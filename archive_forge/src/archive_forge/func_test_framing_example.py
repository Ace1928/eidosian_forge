import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest
@pytest.mark.slow
def test_framing_example():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', 'framing.csv'))
    outcome = np.asarray(data['cong_mesg'])
    outcome_exog = patsy.dmatrix('emo + treat + age + educ + gender + income', data, return_type='dataframe')
    outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=sm.families.links.Probit()))
    mediator = np.asarray(data['emo'])
    mediator_exog = patsy.dmatrix('treat + age + educ + gender + income', data, return_type='dataframe')
    mediator_model = sm.OLS(mediator, mediator_exog)
    tx_pos = [outcome_exog.columns.tolist().index('treat'), mediator_exog.columns.tolist().index('treat')]
    med_pos = outcome_exog.columns.tolist().index('emo')
    med = Mediation(outcome_model, mediator_model, tx_pos, med_pos, outcome_fit_kwargs={'atol': 1e-11})
    np.random.seed(4231)
    para_rslt = med.fit(method='parametric', n_rep=100)
    diff = np.asarray(para_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-06)
    np.random.seed(4231)
    boot_rslt = med.fit(method='boot', n_rep=100)
    diff = np.asarray(boot_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-06)