import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_covariance_type():
    """Test that FisherVectorException is raised when wrong covariance type is
    passed in as a keyword argument.
    """
    with pytest.raises(FisherVectorException):
        learn_gmm(np.random.random((10, 10)), n_modes=2, gm_args={'covariance_type': 'full'})