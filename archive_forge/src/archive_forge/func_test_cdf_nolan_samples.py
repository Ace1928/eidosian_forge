import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
@pytest.mark.parametrize('pct_range,alpha_range,beta_range', [pytest.param([0.01, 0.5, 0.99], [0.1, 1, 2], [-1, 0, 0.8]), pytest.param([0.01, 0.05, 0.5, 0.95, 0.99], [0.1, 0.5, 1, 1.5, 2], [-0.9, -0.5, 0, 0.3, 0.6, 1], marks=pytest.mark.slow), pytest.param([0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 0.95, 0.99], np.linspace(0.1, 2, 20), np.linspace(-1, 1, 21), marks=pytest.mark.xslow)])
def test_cdf_nolan_samples(self, nolan_cdf_sample_data, pct_range, alpha_range, beta_range):
    """ Test cdf values against Nolan's stablec.exe output."""
    data = nolan_cdf_sample_data
    tests = [['piecewise', 2e-12, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & ~((r['alpha'] == 1.0) & np.isin(r['beta'], [-0.3, -0.2, -0.1]) & (r['pct'] == 0.01) | (r['alpha'] == 1.0) & np.isin(r['beta'], [0.1, 0.2, 0.3]) & (r['pct'] == 0.99))], ['piecewise', 0.05, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & ((r['alpha'] == 1.0) & np.isin(r['beta'], [-0.3, -0.2, -0.1]) & (r['pct'] == 0.01)) | (r['alpha'] == 1.0) & np.isin(r['beta'], [0.1, 0.2, 0.3]) & (r['pct'] == 0.99)], ['fft-simpson', 1e-05, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.7)], ['fft-simpson', 0.0001, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.5) & (r['alpha'] <= 1.7)], ['fft-simpson', 0.001, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.3) & (r['alpha'] <= 1.5)], ['fft-simpson', 0.01, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.0) & (r['alpha'] <= 1.3)]]
    for ix, (default_method, rtol, filter_func) in enumerate(tests):
        stats.levy_stable.cdf_default_method = default_method
        subdata = data[filter_func(data)] if filter_func is not None else data
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'Cumulative density calculations experimental for FFT' + ' method. Use piecewise method instead.*')
            p = stats.levy_stable.cdf(subdata['x'], subdata['alpha'], subdata['beta'], scale=1, loc=0)
            with np.errstate(over='ignore'):
                subdata2 = rec_append_fields(subdata, ['calc', 'abserr', 'relerr'], [p, np.abs(p - subdata['p']), np.abs(p - subdata['p']) / np.abs(subdata['p'])])
            failures = subdata2[(subdata2['relerr'] >= rtol) | np.isnan(p)]
            message = f"cdf test {ix} failed with method '{default_method}'\n{failures.dtype.names}\n{failures}"
            assert_allclose(p, subdata['p'], rtol, err_msg=message, verbose=False)