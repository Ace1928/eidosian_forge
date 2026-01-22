import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_patch_extractor_wrong_input(orange_face):
    """Check that an informative error is raised if the patch_size is not valid."""
    faces = _make_images(orange_face)
    err_msg = 'patch_size must be a tuple of two integers'
    extractor = PatchExtractor(patch_size=(8, 8, 8))
    with pytest.raises(ValueError, match=err_msg):
        extractor.transform(faces)