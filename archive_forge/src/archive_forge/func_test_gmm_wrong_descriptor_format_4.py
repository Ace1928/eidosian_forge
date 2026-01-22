import pytest
import numpy as np
from skimage.feature._fisher_vector import (  # noqa: E402
def test_gmm_wrong_descriptor_format_4():
    """Test that DescriptorException is raised when elements of descriptor list
    are of the incorrect type (i.e. not a NumPy ndarray).
    """
    with pytest.raises(DescriptorException):
        learn_gmm([[1, 2, 3], [1, 2, 3]], n_modes=1)