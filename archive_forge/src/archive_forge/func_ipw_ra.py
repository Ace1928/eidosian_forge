import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
@Substitution(params_returns=indent(doc_params_returns, ' ' * 8))
def ipw_ra(self, return_results=True, effect_group='all', disp=False):
    """
        ATE and POM from inverse probability weighted regression adjustment.

        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
    treat_mask = self.treat_mask
    endog = self.model_pool.endog
    exog = self.model_pool.exog
    prob = self.prob_select
    prob0 = prob[~treat_mask]
    prob1 = prob[treat_mask]
    if effect_group == 'all':
        w0 = 1 / (1 - prob0)
        w1 = 1 / prob1
        exogt = exog
    elif effect_group in [1, 'treated']:
        w0 = prob0 / (1 - prob0)
        w1 = prob1 / prob1
        exogt = exog[treat_mask]
        effect_group = 1
    elif effect_group in [0, 'untreated', 'control']:
        w0 = (1 - prob0) / (1 - prob0)
        w1 = (1 - prob1) / prob1
        exogt = exog[~treat_mask]
        effect_group = 0
    else:
        raise ValueError('incorrect option for effect_group')
    mod0 = WLS(endog[~treat_mask], exog[~treat_mask], weights=w0)
    result0 = mod0.fit(cov_type='HC1')
    mean0_ipwra = result0.predict(exogt).mean()
    mod1 = WLS(endog[treat_mask], exog[treat_mask], weights=w1)
    result1 = mod1.fit(cov_type='HC1')
    mean1_ipwra = result1.predict(exogt).mean()
    if not return_results:
        return (mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra)
    mod_gmm = _IPWRAGMM(endog, self.results_select, None, teff=self, effect_group=effect_group)
    start_params = np.concatenate(([mean1_ipwra - mean0_ipwra, mean0_ipwra], result0.params, result1.params, np.asarray(self.results_select.params)))
    res_gmm = mod_gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 2000, 'disp': disp}, maxiter=1)
    res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group=effect_group)
    return res