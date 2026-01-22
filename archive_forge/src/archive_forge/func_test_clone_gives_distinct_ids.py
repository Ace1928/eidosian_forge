import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_clone_gives_distinct_ids(nH, nI):
    model = clone(Linear(nH), 5)
    assert len(model.layers) == 5
    seen_ids = set()
    for node in model.walk():
        assert node.id not in seen_ids
        seen_ids.add(node.id)
    assert len(seen_ids) == 6