import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
@pytest.fixture
def downsampled_face_collection(downsampled_face):
    return _make_images(downsampled_face)