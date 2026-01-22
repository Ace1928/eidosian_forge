import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao
def res_opg(self):
    res_ols = self.res_ols
    nobs = self.nobs
    moms = self.moms
    moms_obs = self.moms_obs
    covm = self.covm
    moms_deriv = self.moms_deriv
    weights = self.weights
    L = self.L
    x = self.exog_full
    res_ols2_hc0 = OLS(res_ols.model.endog, x).fit(cov_type='HC0')
    res_all = []
    ones = np.ones(nobs)
    stat = nobs * OLS(ones, moms_obs).fit().rsquared
    res_all.append(('ols R2', stat))
    tres = res_ols2_hc0.compare_lm_test(res_ols, demean=False)
    res_all.append(('comp_lm uc', tres))
    tres = CMTNewey(moms, covm, covm[:, :-2], weights, L).chisquare
    res_all.append(('Newey', tres))
    tres = CMTTauchen(moms[:-2], covm[:-2, :-2], moms[-2:], covm[-2:, :-2], covm).chisquare
    res_all.append(('Tauchen', tres))
    tres = diao.lm_robust_subset(moms[-2:], 2, covm, covm)
    res_all.append(('score subset QMLE', tres))
    tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:], np.linalg.inv(covm), covm, cov_params=None)
    res_all.append(('scoreB QMLE', tres))
    tres = diao.lm_robust(moms, np.eye(moms.shape[0])[-2:], np.linalg.inv(covm), None, cov_params=np.linalg.inv(covm))
    res_all.append(('scoreV QMLE', tres))
    return res_all