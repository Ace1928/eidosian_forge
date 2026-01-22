import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_extract_patches_all_rect(downsampled_face):
    face = downsampled_face
    face = face[:, 32:97]
    i_h, i_w = face.shape
    p_h, p_w = (16, 12)
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)
    patches = extract_patches_2d(face, (p_h, p_w))
    assert patches.shape == (expected_n_patches, p_h, p_w)