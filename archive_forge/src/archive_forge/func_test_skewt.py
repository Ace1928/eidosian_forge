import numpy as np
from numpy.testing import assert_, assert_almost_equal
from statsmodels.sandbox.distributions.extras import (skewnorm,
def test_skewt():
    skewt = ACSkewT_gen()
    x = [-2, -1, -0.5, 0, 1, 2]
    pdf_r = np.array([2.973416551551523e-90, 3.687562713971017e-24, 2.01840158642297e-07, 0.3989422804014327, 0.4839414490382867, 0.1079819330263761])
    pdf_st = skewt.pdf(x, 1000000, 10)
    pass
    np.allclose(pdf_st, pdf_r, rtol=0, atol=1e-06)
    np.allclose(pdf_st, pdf_r, rtol=0.1, atol=0)
    cdf_r = np.array([0.0, 0.0, 3.729478836866917e-09, 0.03172551743055357, 0.6826894921370859, 0.9544997361036416])
    cdf_st = skewt.cdf(x, 1000000, 10)
    np.allclose(cdf_st, cdf_r, rtol=0, atol=1e-06)
    np.allclose(cdf_st, cdf_r, rtol=0.1, atol=0)
    pdf_r = np.array([2.185448836190663e-07, 1.272381597868587e-05, 0.0005746937644959992, 0.3796066898224945, 0.4393468708859825, 0.1301804021075493])
    pdf_st = skewt.pdf(x, 5, 10)
    assert_(np.allclose(pdf_st, pdf_r, rtol=1e-13, atol=1e-25))
    cdf_r = np.array([8.822783669199699e-08, 2.638467463775795e-06, 6.573106017198583e-05, 0.03172551743055352, 0.6367851708183412, 0.8980606093979784])
    cdf_st = skewt.cdf(x, 5, 10)
    assert_(np.allclose(cdf_st, cdf_r, rtol=1e-10, atol=0))
    pdf_r = np.array([0.0003941955996757291, 0.001568067236862745, 0.006136996029432048, 0.3183098861837907, 0.3167418189469279, 0.1269297588738406])
    pdf_st = skewt.pdf(x, 1, 10)
    assert_(np.allclose(pdf_st, pdf_r, rtol=1e-13, atol=1e-25))
    cdf_r = np.array([0.0007893671370544414, 0.001575817262600422, 0.00312872074910556, 0.03172551743055351, 0.5015758172626005, 0.7056221318361879])
    cdf_st = skewt.cdf(x, 1, 10)
    assert_(np.allclose(cdf_st, cdf_r, rtol=1e-13, atol=1e-25))