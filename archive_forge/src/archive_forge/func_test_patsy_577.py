import numpy as np
import pandas
from statsmodels.tools import data
def test_patsy_577():
    X = np.random.random((10, 2))
    df = pandas.DataFrame(X, columns=['var1', 'var2'])
    from patsy import dmatrix
    endog = dmatrix('var1 - 1', df)
    np.testing.assert_(data._is_using_patsy(endog, None))
    exog = dmatrix('var2 - 1', df)
    np.testing.assert_(data._is_using_patsy(endog, exog))