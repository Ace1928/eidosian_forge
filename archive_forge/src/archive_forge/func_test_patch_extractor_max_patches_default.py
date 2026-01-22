import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_patch_extractor_max_patches_default(downsampled_face_collection):
    faces = downsampled_face_collection
    extr = PatchExtractor(max_patches=100, random_state=0)
    patches = extr.transform(faces)
    assert patches.shape == (len(faces) * 100, 19, 25)