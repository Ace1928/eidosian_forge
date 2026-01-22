import numpy as np
from statsmodels.tsa.interp import dentonm
def test_denton_quarterly():
    indicator = np.array([98.2, 100.8, 102.2, 100.8, 99.0, 101.6, 102.7, 101.5, 100.5, 103.0, 103.5, 101.5])
    benchmark = np.array([4000.0, 4161.4])
    x_imf = dentonm(indicator, benchmark, freq='aq')
    imf_stata = np.array([969.8, 998.4, 1018.3, 1013.4, 1007.2, 1042.9, 1060.3, 1051.0, 1040.6, 1066.5, 1071.7, 1051.0])
    np.testing.assert_almost_equal(imf_stata, x_imf, 1)