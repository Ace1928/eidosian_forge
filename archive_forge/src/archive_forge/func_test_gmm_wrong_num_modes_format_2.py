import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_num_modes_format_2():
    """Test that FisherVectorException is raised when a number that is not a
    positive integer is passed into the n_modes argument of learn_gmm.
    """
    with pytest.raises(FisherVectorException):
        learn_gmm([np.zeros((5, 10)), np.zeros((4, 10))], n_modes=-1)