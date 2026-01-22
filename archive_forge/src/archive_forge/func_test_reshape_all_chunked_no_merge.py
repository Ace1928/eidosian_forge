from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.reshape import contract_tuple, expand_tuple, reshape_rechunk
from dask.array.utils import assert_eq
@pytest.mark.parametrize('inshape, inchunks, outshape, outchunks', [((2, 3, 4), ((1, 1), (1, 2), (2, 2)), (6, 4), ((1, 2, 1, 2), (2, 2))), ((1, 2, 3, 4), ((1,), (1, 1), (1, 2), (2, 2)), (6, 4), ((1, 2, 1, 2), (2, 2))), ((2, 2, 3, 4), ((1, 1), (1, 1), (1, 2), (2, 2)), (12, 4), ((1, 2, 1, 2, 1, 2, 1, 2), (2, 2))), ((2, 2, 3, 4), ((1, 1), (1, 1), (1, 2), (2, 2)), (4, 3, 4), ((1, 1, 1, 1), (1, 2), (2, 2))), ((2, 2, 3, 4), ((1, 1), (2,), (1, 2), (4,)), (4, 3, 4), ((2, 2), (1, 2), (4,))), ((2, 3, 4), ((1, 1), (1, 1, 1), (2, 2)), (24,), ((2,) * 12,)), ((2, 3, 4), ((1, 1), (1, 1, 1), (4,)), (2, 12), ((1, 1), (4,) * 3))])
def test_reshape_all_chunked_no_merge(inshape, inchunks, outshape, outchunks):
    base = np.arange(np.prod(inshape)).reshape(inshape)
    a = da.from_array(base, chunks=inchunks)
    inchunks2, outchunks2 = reshape_rechunk(a.shape, outshape, inchunks)
    assert inchunks2 == inchunks
    assert outchunks2 == outchunks
    result = a.reshape(outshape)
    assert result.chunks == outchunks
    assert_eq(result, base.reshape(outshape))