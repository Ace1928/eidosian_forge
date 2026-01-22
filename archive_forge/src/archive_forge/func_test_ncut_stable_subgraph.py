import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_ncut_stable_subgraph():
    """Test to catch an error thrown when subgraph has all equal edges."""
    img = np.zeros((100, 100, 3), dtype='uint8')
    labels = np.zeros((100, 100), dtype='uint8')
    labels[:50, :50] = 1
    labels[:50, 50:] = 2
    rag = graph.rag_mean_color(img, labels, mode='similarity')
    new_labels = graph.cut_normalized(labels, rag, in_place=False)
    new_labels, _, _ = segmentation.relabel_sequential(new_labels)
    assert new_labels.max() == 0