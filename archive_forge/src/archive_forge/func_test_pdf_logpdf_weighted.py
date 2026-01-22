from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_pdf_logpdf_weighted():
    np.random.seed(1)
    n_basesample = 50
    xn = np.random.randn(n_basesample)
    wn = np.random.rand(n_basesample)
    gkde = stats.gaussian_kde(xn, weights=wn)
    xs = np.linspace(-15, 12, 25)
    pdf = gkde.evaluate(xs)
    pdf2 = gkde.pdf(xs)
    assert_almost_equal(pdf, pdf2, decimal=12)
    logpdf = np.log(pdf)
    logpdf2 = gkde.logpdf(xs)
    assert_almost_equal(logpdf, logpdf2, decimal=12)
    gkde = stats.gaussian_kde(xs, weights=np.random.rand(len(xs)))
    pdf = np.log(gkde.evaluate(xn))
    pdf2 = gkde.logpdf(xn)
    assert_almost_equal(pdf, pdf2, decimal=12)