import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
@pytest.mark.parametrize('outdt', [None, np.float32, np.float64])
@pytest.mark.parametrize('filter', _3d_rank_filters)
def test_rank_filters_3D(self, filter, outdt):

    @run_in_parallel(warnings_matching=['Possible precision loss'])
    def check():
        expected = self.refs_3d[filter]
        if outdt is not None:
            out = np.zeros_like(expected, dtype=outdt)
        else:
            out = None
        result = getattr(rank, filter)(self.volume, self.footprint_3d, out=out)
        if outdt is not None:
            if filter == 'sum':
                datadt = np.uint8
            else:
                datadt = expected.dtype
            result = np.mod(result, 256.0).astype(datadt)
        assert_array_almost_equal(expected, result)
    check()