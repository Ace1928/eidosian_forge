import numpy
import pytest
from thinc import registry
from thinc.api import (
def test_cosine_orthogonal():
    vec1 = numpy.asarray([[0, 2], [0, 5]])
    vec2 = numpy.asarray([[8, 0], [7, 0]])
    d_vecs = CosineDistance(normalize=True).get_grad(vec1, vec2)
    assert d_vecs.shape == vec1.shape
    assert d_vecs[0][0] < 0
    assert d_vecs[0][1] > 0
    assert d_vecs[1][0] < 0
    assert d_vecs[1][1] > 0
    loss_not_normalized = CosineDistance(normalize=False).get_loss(vec1, vec2)
    assert loss_not_normalized == pytest.approx(2, eps)
    loss_normalized = CosineDistance(normalize=True).get_loss(vec1, vec2)
    assert loss_normalized == pytest.approx(1, eps)