import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_connect_regions_with_grid(raccoon_face_fxt):
    face = raccoon_face_fxt
    face = face[::4, ::4]
    mask = face > 50
    graph = grid_to_graph(*face.shape, mask=mask)
    assert ndimage.label(mask)[1] == connected_components(graph)[0]
    mask = face > 150
    graph = grid_to_graph(*face.shape, mask=mask, dtype=None)
    assert ndimage.label(mask)[1] == connected_components(graph)[0]