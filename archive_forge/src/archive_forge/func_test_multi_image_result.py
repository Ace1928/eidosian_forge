from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
def test_multi_image_result(self):
    source = ogc.WMSRasterSource(self.URI, self.layer)
    crs = ccrs.PlateCarree(central_longitude=180)
    extent = [-15, 25, 45, 85]
    located_images = source.fetch_raster(crs, extent, RESOLUTION)
    assert len(located_images) == 2