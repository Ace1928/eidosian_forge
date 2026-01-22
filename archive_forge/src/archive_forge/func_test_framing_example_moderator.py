import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest
def test_framing_example_moderator():
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
    ix = (outcome_exog.columns.tolist().index('age'), mediator_exog.columns.tolist().index('age'))
    moderators = {ix: 20}
    med = Mediation(outcome_model, mediator_model, tx_pos, med_pos, moderators=moderators)
    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=100)