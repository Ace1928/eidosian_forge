import numpy
import pytest
from thinc.layers import premap_ids, remap_ids, remap_ids_v2
def test_column(keys, mapper):
    idx = numpy.zeros((len(keys), 4), dtype='int')
    idx[:, 3] = keys
    remap_v2 = remap_ids_v2(mapper, column=3)
    premap = premap_ids(mapper, column=3)
    numpy.testing.assert_equal(remap_v2(idx, False)[0].squeeze(), numpy.asarray(range(len(keys))))
    numpy.testing.assert_equal(premap(idx, False)[0].squeeze(), numpy.asarray(range(len(keys))))