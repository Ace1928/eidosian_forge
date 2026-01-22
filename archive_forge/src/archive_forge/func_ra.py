import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
@Substitution(params_returns=indent(doc_params_returns, ' ' * 8))
def ra(self, return_results=True, effect_group='all', disp=False):
    """
        Regression Adjustment treatment effect estimation.
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults
        """
    tind = np.zeros(len(self.treatment))
    tind[-self.treatment.sum():] = 1
    if effect_group == 'all':
        probt = None
    elif effect_group in [1, 'treated']:
        probt = tind
        effect_group = 1
    elif effect_group in [0, 'untreated', 'control']:
        probt = 1 - tind
        effect_group = 0
    elif isinstance(effect_group, np.ndarray):
        probt = effect_group
        effect_group = 'user'
    else:
        raise ValueError('incorrect option for effect_group')
    exog = self.exog_grouped
    if probt is not None:
        cw = probt / probt.mean()
    else:
        cw = 1
    pom0 = (self.results0.predict(exog) * cw).mean()
    pom1 = (self.results1.predict(exog) * cw).mean()
    if not return_results:
        return (pom1 - pom0, pom0, pom1)
    endog = self.model_pool.endog
    mod_gmm = _RAGMM(endog, self.results_select, None, teff=self, probt=probt)
    start_params = np.concatenate(([pom1 - pom0, pom0], self.results0.params, self.results1.params))
    res_gmm = mod_gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
    res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group=effect_group)
    return res