import cartopy.crs as ccrs
import pytest
import geoviews as gv
from geoviews.util import from_xarray, process_crs
@pytest.mark.skipif(rxr is None, reason='Needs rioxarray to be installed')
def test_from_xarray():
    file = 'https://github.com/holoviz/hvplot/raw/main/hvplot/tests/data/RGB-red.byte.tif'
    output = from_xarray(rxr.open_rasterio(file))
    assert isinstance(output, gv.RGB)
    assert sorted(map(str, output.kdims)) == ['x', 'y']
    assert isinstance(output.crs, ccrs.CRS)