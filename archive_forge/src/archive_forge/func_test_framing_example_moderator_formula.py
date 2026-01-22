import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest
@pytest.mark.slow
def test_framing_example_moderator_formula():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', 'framing.csv'))
    outcome_model = sm.GLM.from_formula('cong_mesg ~ emo + treat*age + emo*age + educ + gender + income', data, family=sm.families.Binomial(link=sm.families.links.Probit()))
    mediator_model = sm.OLS.from_formula('emo ~ treat*age + educ + gender + income', data)
    moderators = {'age': 20}
    med = Mediation(outcome_model, mediator_model, 'treat', 'emo', moderators=moderators)
    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=100)
    diff = np.asarray(med_rslt.summary() - framing_moderated_4231)
    assert_allclose(diff, 0, atol=1e-06)