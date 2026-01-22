import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_all_samples_same_location_bw(self):
    x = np.ones(100)
    kde = nparam.KDEUnivariate(x)
    with pytest.raises(RuntimeError, match='Selected KDE bandwidth is 0'):
        kde.fit()