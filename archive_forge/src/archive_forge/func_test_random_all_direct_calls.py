from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
@pytest.mark.parametrize('backend', ['cupy', 'numpy'])
def test_random_all_direct_calls(backend):
    if backend == 'cupy':
        expect = cupy.ndarray
    else:
        expect = np.ndarray

    def rnd_test(func, *args, **kwargs):
        a = func(*args, **kwargs)
        assert type(a._meta) == expect
        assert_eq(a, a)
    with config.set({'array.backend': backend}):
        rnd_test(da.random.beta, 1, 2, size=5, chunks=3)
        rnd_test(da.random.binomial, 10, 0.5, size=5, chunks=3)
        rnd_test(da.random.chisquare, 1, size=5, chunks=3)
        rnd_test(da.random.exponential, 1, size=5, chunks=3)
        rnd_test(da.random.f, 1, 2, size=5, chunks=3)
        rnd_test(da.random.gamma, 5, 1, size=5, chunks=3)
        rnd_test(da.random.geometric, 1, size=5, chunks=3)
        rnd_test(da.random.gumbel, 1, size=5, chunks=3)
        rnd_test(da.random.hypergeometric, 1, 2, 3, size=5, chunks=3)
        rnd_test(da.random.laplace, size=5, chunks=3)
        rnd_test(da.random.logistic, size=5, chunks=3)
        rnd_test(da.random.lognormal, size=5, chunks=3)
        rnd_test(da.random.logseries, 0.5, size=5, chunks=3)
        if backend != 'cupy':
            rnd_test(da.random.multinomial, 20, [1 / 6.0] * 6, size=5, chunks=3)
        rnd_test(da.random.negative_binomial, 5, 0.5, size=5, chunks=3)
        rnd_test(da.random.noncentral_chisquare, 2, 2, size=5, chunks=3)
        rnd_test(da.random.noncentral_f, 2, 2, 3, size=5, chunks=3)
        rnd_test(da.random.normal, 2, 2, size=5, chunks=3)
        rnd_test(da.random.pareto, 1, size=5, chunks=3)
        rnd_test(da.random.poisson, size=5, chunks=3)
        rnd_test(da.random.power, 1, size=5, chunks=3)
        rnd_test(da.random.rayleigh, size=5, chunks=3)
        rnd_test(da.random.randint, low=10, size=5, chunks=3)
        rnd_test(da.random.random, size=5, chunks=3)
        rnd_test(da.random.random_sample, size=5, chunks=3)
        rnd_test(da.random.triangular, 1, 2, 3, size=5, chunks=3)
        rnd_test(da.random.uniform, size=5, chunks=3)
        rnd_test(da.random.vonmises, 2, 3, size=5, chunks=3)
        rnd_test(da.random.wald, 1, 2, size=5, chunks=3)
        rnd_test(da.random.weibull, 2, size=5, chunks=3)
        rnd_test(da.random.zipf, 2, size=5, chunks=3)
        rnd_test(da.random.standard_cauchy, size=5, chunks=3)
        rnd_test(da.random.standard_exponential, size=5, chunks=3)
        rnd_test(da.random.standard_gamma, 2, size=5, chunks=3)
        rnd_test(da.random.standard_normal, size=5, chunks=3)
        rnd_test(da.random.standard_t, 2, size=5, chunks=3)