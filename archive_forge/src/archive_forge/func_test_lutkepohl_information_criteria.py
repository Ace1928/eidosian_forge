from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_lutkepohl_information_criteria():
    dta = pd.DataFrame(results_var_misc.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()
    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
    true = results_var_misc.lutkepohl_ar1_lustats
    mod = sarimax.SARIMAX(endog['dln_inv'], order=(1, 0, 0), trend='c', loglikelihood_burn=1)
    res = mod.filter(true['params'])
    assert_allclose(res.llf, true['loglike'])
    aic = res.info_criteria('aic', method='lutkepohl') - 2 * 2 / res.nobs_effective
    bic = res.info_criteria('bic', method='lutkepohl') - 2 * np.log(res.nobs_effective) / res.nobs_effective
    hqic = res.info_criteria('hqic', method='lutkepohl') - 2 * 2 * np.log(np.log(res.nobs_effective)) / res.nobs_effective
    assert_allclose(aic, true['aic'])
    assert_allclose(bic, true['bic'])
    assert_allclose(hqic, true['hqic'])
    true = results_var_misc.lutkepohl_ar1
    aic = res.aic - 2
    bic = res.bic - np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])
    aic = res.info_criteria('aic') - 2
    bic = res.info_criteria('bic') - np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])
    true = results_var_misc.lutkepohl_var1_lustats
    mod = varmax.VARMAX(endog, order=(1, 0), trend='n', error_cov_type='unstructured', loglikelihood_burn=1)
    res = mod.filter(true['params'])
    assert_allclose(res.llf, true['loglike'])
    aic = res.info_criteria('aic', method='lutkepohl') - 2 * 6 / res.nobs_effective
    bic = res.info_criteria('bic', method='lutkepohl') - 6 * np.log(res.nobs_effective) / res.nobs_effective
    hqic = res.info_criteria('hqic', method='lutkepohl') - 2 * 6 * np.log(np.log(res.nobs_effective)) / res.nobs_effective
    assert_allclose(aic, true['aic'])
    assert_allclose(bic, true['bic'])
    assert_allclose(hqic, true['hqic'])
    true = results_var_misc.lutkepohl_var1
    aic = res.aic - 2 * 6
    bic = res.bic - 6 * np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])
    aic = res.info_criteria('aic') - 2 * 6
    bic = res.info_criteria('bic') - 6 * np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])