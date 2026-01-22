import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
def run_ucm(name, use_exact_diffuse=False):
    true = getattr(results_structural, name)
    for model in true['models']:
        kwargs = model.copy()
        kwargs.update(true['kwargs'])
        kwargs['use_exact_diffuse'] = use_exact_diffuse
        values = dta.copy()
        freq = kwargs.pop('freq', None)
        if freq is not None:
            values.index = pd.date_range(start='1959-01-01', periods=len(dta), freq=freq)
        if 'exog' in kwargs:
            exog = np.log(values['realgdp'])
            if kwargs['exog'] == 'numpy':
                exog = exog.values.squeeze()
            kwargs['exog'] = exog
        mod = UnobservedComponents(values['unemp'], **kwargs)
        mod.start_params
        roundtrip = mod.transform_params(mod.untransform_params(mod.start_params))
        assert_allclose(mod.start_params, roundtrip)
        res_true = mod.filter(true['params'])
        freqstr = freq[0] if freq is not None else values.index.freqstr[0]
        if 'cycle_period_bounds' in kwargs:
            cycle_period_bounds = kwargs['cycle_period_bounds']
        elif freqstr in ('A', 'AS', 'Y', 'YS'):
            cycle_period_bounds = (1.5, 12)
        elif freqstr in ('Q', 'QS'):
            cycle_period_bounds = (1.5 * 4, 12 * 4)
        elif freqstr in ('M', 'MS'):
            cycle_period_bounds = (1.5 * 12, 12 * 12)
        else:
            cycle_period_bounds = (2, np.inf)
        assert_equal(mod.cycle_frequency_bound, (2 * np.pi / cycle_period_bounds[1], 2 * np.pi / cycle_period_bounds[0]))
        rtol = true.get('rtol', 1e-07)
        atol = true.get('atol', 0)
        if use_exact_diffuse:
            res_llf = res_true.llf_obs.sum() + res_true.nobs_diffuse * 0.5 * np.log(2 * np.pi)
        else:
            res_llf = res_true.llf_obs[res_true.loglikelihood_burn:].sum()
        assert_allclose(res_llf, true['llf'], rtol=rtol, atol=atol)
        try:
            import matplotlib.pyplot as plt
            try:
                from pandas.plotting import register_matplotlib_converters
                register_matplotlib_converters()
            except ImportError:
                pass
            fig = plt.figure()
            res_true.plot_components(fig=fig)
        except ImportError:
            pass
        with warnings.catch_warnings(record=True):
            fit_kwargs = {}
            if 'maxiter' in true:
                fit_kwargs['maxiter'] = true['maxiter']
            res = mod.fit(start_params=true.get('start_params', None), disp=-1, **fit_kwargs)
            if use_exact_diffuse:
                res_llf = res.llf_obs.sum() + res.nobs_diffuse * 0.5 * np.log(2 * np.pi)
            else:
                res_llf = res.llf_obs[res_true.loglikelihood_burn:].sum()
            if res_llf <= true['llf']:
                assert_allclose(res_llf, true['llf'], rtol=0.0001)
            res.summary()