import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
def test_descriptor_extractor():
    with pytest.raises(NotImplementedError):
        DescriptorExtractor().extract(None, None)