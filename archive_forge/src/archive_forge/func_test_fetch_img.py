from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
def test_fetch_img(self):
    source = ogc.WMTSRasterSource(self.URI, self.layer_name)
    extent = [-10, 10, 40, 60]
    located_image, = source.fetch_raster(self.projection, extent, RESOLUTION)
    img = np.array(located_image.image)
    assert img.shape == (512, 512, 4)
    assert img[:, :, 3].min() == 255
    assert located_image.extent == (-180.0, 107.99999999999994, -197.99999999999994, 90.0)