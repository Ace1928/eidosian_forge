import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import array
def exampletest(res_armarep):
    from statsmodels.sandbox import tsa
    arrep = tsa.arma_impulse_response(res_armarep.ma, res_armarep.ar, nobs=21)[1:]
    marep = tsa.arma_impulse_response(res_armarep.ar, res_armarep.ma, nobs=21)[1:]
    assert_array_almost_equal(res_armarep.marep.ravel(), marep, 14)
    assert_array_almost_equal(-res_armarep.arrep.ravel(), arrep, 14)