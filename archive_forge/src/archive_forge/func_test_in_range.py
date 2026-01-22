import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
def test_in_range(self, Source):
    if Source == cartopy.io.srtm.SRTM3Source:
        shape = (1201, 1201)
    elif Source == cartopy.io.srtm.SRTM1Source:
        shape = (3601, 3601)
    else:
        raise ValueError('Source is of unexpected type.')
    source = Source()
    img, crs, extent = source.single_tile(-1, 50)
    assert isinstance(img, np.ndarray)
    assert img.shape == shape
    assert img.dtype == np.dtype('>i2')
    assert crs == ccrs.PlateCarree()
    assert extent == (-1, 0, 50, 51)