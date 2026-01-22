import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_patch_extractor_max_patches(downsampled_face_collection):
    faces = downsampled_face_collection
    i_h, i_w = faces.shape[1:3]
    p_h, p_w = (8, 8)
    max_patches = 100
    expected_n_patches = len(faces) * max_patches
    extr = PatchExtractor(patch_size=(p_h, p_w), max_patches=max_patches, random_state=0)
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w)
    max_patches = 0.5
    expected_n_patches = len(faces) * int((i_h - p_h + 1) * (i_w - p_w + 1) * max_patches)
    extr = PatchExtractor(patch_size=(p_h, p_w), max_patches=max_patches, random_state=0)
    patches = extr.transform(faces)
    assert patches.shape == (expected_n_patches, p_h, p_w)