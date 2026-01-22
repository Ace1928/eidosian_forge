import numpy
import pytest
from thinc.layers import premap_ids, remap_ids, remap_ids_v2
def test_premap(keys, mapper):
    premap = premap_ids(mapper, default=99)
    values, _ = premap(keys, False)
    numpy.testing.assert_equal(values.squeeze(), numpy.asarray(range(len(keys))))