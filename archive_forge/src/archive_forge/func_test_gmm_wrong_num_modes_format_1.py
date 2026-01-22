import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_num_modes_format_1():
    """Test that FisherVectorException is raised when incorrect type for
    n_modes is passed into the learn_gmm function.
    """
    with pytest.raises(FisherVectorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10))], n_modes='not_valid')