import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def compare_r_output_dogs_data(method):
    """ Testing within-subject effect interact with 2 between-subject effect
    Compares with R car library Anova(, type=3) output

    Note: The test statistis Phillai, Wilks, Hotelling-Lawley
          and Roy are the same as R output but the approximate F and degree
          of freedoms can be different. This is due to the fact that this
          implementation is based on SAS formula [1]

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    """
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method=method)
    r = r.mv_test()
    a = [[0.026860766, 4, 6, 54.3435304, 7.5958561e-05], [0.973139234, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05]]
    assert_array_almost_equal(r['Intercept']['stat'].values, a, decimal=6)
    a = [[0.0839646619, 8, 12.0, 3.67658068, 0.0212614444], [1.18605382, 8, 14.0, 2.55003861, 0.0601270701], [7.69391362, 8, 6.63157895, 5.5081427, 0.020739226], [7.25036952, 4, 7.0, 12.6881467, 0.00252669877]]
    assert_array_almost_equal(r['Drug']['stat'].values, a, decimal=6)
    a = [[0.32048892, 4.0, 6.0, 3.18034906, 0.10002373], [0.67951108, 4.0, 6.0, 3.18034906, 0.10002373], [2.12023271, 4.0, 6.0, 3.18034906, 0.10002373], [2.12023271, 4.0, 6.0, 3.18034906, 0.10002373]]
    assert_array_almost_equal(r['Depleted']['stat'].values, a, decimal=6)
    a = [[0.15234366, 8.0, 12.0, 2.34307678, 0.08894239], [1.13013353, 8.0, 14.0, 2.27360606, 0.08553213], [3.70989596, 8.0, 6.63157895, 2.65594824, 0.11370285], [3.1145597, 4.0, 7.0, 5.45047947, 0.02582767]]
    assert_array_almost_equal(r['Drug:Depleted']['stat'].values, a, decimal=6)