import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_extract_patches_less_than_max_patches(downsampled_face):
    face = downsampled_face
    i_h, i_w = face.shape
    p_h, p_w = (3 * i_h // 4, 3 * i_w // 4)
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)
    patches = extract_patches_2d(face, (p_h, p_w), max_patches=4000)
    assert patches.shape == (expected_n_patches, p_h, p_w)