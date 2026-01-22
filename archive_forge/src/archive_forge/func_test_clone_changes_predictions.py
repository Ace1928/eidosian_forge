import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_clone_changes_predictions(nH, nI):
    model1 = Linear(nH)
    model = clone(model1, 10)
    ones = numpy.ones((10, nI), dtype='f')
    model.initialize(X=ones)
    output_from_cloned = model.predict(ones)
    output_from_orig = model1.predict(ones)
    assert output_from_cloned.sum() != output_from_orig.sum()