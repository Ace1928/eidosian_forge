import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
@pytest.mark.parametrize('sn, n_fit_samples', [(500, 200)])
def test_cont_basic(distname, arg, sn, n_fit_samples):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'
    rng = np.random.RandomState(765456)
    rvs = distfn.rvs(*arg, size=sn, random_state=rng)
    m, v = distfn.stats(*arg)
    if distname not in {'laplace_asymmetric'}:
        check_sample_meanvar_(m, v, rvs)
    check_cdf_ppf(distfn, arg, distname)
    check_sf_isf(distfn, arg, distname)
    check_cdf_sf(distfn, arg, distname)
    check_ppf_isf(distfn, arg, distname)
    check_pdf(distfn, arg, distname)
    check_pdf_logpdf(distfn, arg, distname)
    check_pdf_logpdf_at_endpoints(distfn, arg, distname)
    check_cdf_logcdf(distfn, arg, distname)
    check_sf_logsf(distfn, arg, distname)
    check_ppf_broadcast(distfn, arg, distname)
    alpha = 0.01
    if distname == 'rv_histogram_instance':
        check_distribution_rvs(distfn.cdf, arg, alpha, rvs)
    elif distname != 'geninvgauss':
        check_distribution_rvs(distname, arg, alpha, rvs)
    locscale_defaults = (0, 1)
    meths = [distfn.pdf, distfn.logpdf, distfn.cdf, distfn.logcdf, distfn.logsf]
    spec_x = {'weibull_max': -0.5, 'levy_l': -0.5, 'pareto': 1.5, 'truncpareto': 3.2, 'tukeylambda': 0.3, 'rv_histogram_instance': 5.0}
    x = spec_x.get(distname, 0.5)
    if distname == 'invweibull':
        arg = (1,)
    elif distname == 'ksone':
        arg = (3,)
    check_named_args(distfn, x, arg, locscale_defaults, meths)
    check_random_state_property(distfn, arg)
    if distname in ['rel_breitwigner'] and _IS_32BIT:
        pytest.skip('fails on Linux 32-bit')
    else:
        check_pickling(distfn, arg)
    check_freezing(distfn, arg)
    if distname not in ['kstwobign', 'kstwo', 'ncf']:
        check_entropy(distfn, arg, distname)
    if distfn.numargs == 0:
        check_vecentropy(distfn, arg)
    if distfn.__class__._entropy != stats.rv_continuous._entropy and distname != 'vonmises':
        check_private_entropy(distfn, arg, stats.rv_continuous)
    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The occurrence of roundoff error')
        sup.filter(IntegrationWarning, 'Extremely bad integrand')
        sup.filter(RuntimeWarning, 'invalid value')
        check_entropy_vect_scale(distfn, arg)
    check_retrieving_support(distfn, arg)
    check_edge_support(distfn, arg)
    check_meth_dtype(distfn, arg, meths)
    check_ppf_dtype(distfn, arg)
    if distname not in fails_cmplx:
        check_cmplx_deriv(distfn, arg)
    if distname != 'truncnorm':
        check_ppf_private(distfn, arg, distname)
    for method in ['MLE', 'MM']:
        if distname not in skip_fit_test[method]:
            check_fit_args(distfn, arg, rvs[:n_fit_samples], method)
        if distname not in skip_fit_fix_test[method]:
            check_fit_args_fix(distfn, arg, rvs[:n_fit_samples], method)