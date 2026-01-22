import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
@pytest.mark.parametrize('Source, shape', [(cartopy.io.srtm.SRTM3Source, (1201, 1201)), (cartopy.io.srtm.SRTM1Source, (3601, 3601))], ids=['srtm3', 'srtm1'])
def test_srtm_out_of_range(self, Source, shape):
    img, _, _ = Source().combined(120, 2, 2, 2)
    assert_array_equal(img, np.zeros(np.array(shape) * 2))