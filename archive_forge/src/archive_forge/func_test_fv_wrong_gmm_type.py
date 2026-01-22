import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_fv_wrong_gmm_type():
    """
    Test that FisherVectorException is raised when a GMM not of type
    sklearn.mixture.GaussianMixture is passed into the fisher_vector
    function.
    """

    class MyDifferentGaussianMixture:
        pass
    with pytest.raises(FisherVectorException):
        fisher_vector(np.zeros((10, 10)), MyDifferentGaussianMixture())