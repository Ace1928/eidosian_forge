import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
@pytest.fixture
def orange_face(downsampled_face):
    face = downsampled_face
    face_color = np.zeros(face.shape + (3,))
    face_color[:, :, 0] = 256 - face
    face_color[:, :, 1] = 256 - face / 2
    face_color[:, :, 2] = 256 - face / 4
    return face_color