import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_connect_regions(raccoon_face_fxt):
    face = raccoon_face_fxt
    face = face[::4, ::4]
    for thr in (50, 150):
        mask = face > thr
        graph = img_to_graph(face, mask=mask)
        assert ndimage.label(mask)[1] == connected_components(graph)[0]