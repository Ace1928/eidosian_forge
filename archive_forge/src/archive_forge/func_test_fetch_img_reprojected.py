from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
def test_fetch_img_reprojected(self):
    source = ogc.WMTSRasterSource(self.URI, self.layer_name)
    extent = [-20, -1, 48, 50]
    located_image, = source.fetch_raster(ccrs.NorthPolarStereo(), extent, (30, 30))
    img = np.array(located_image.image)
    assert img.shape == (42, 42, 4)
    assert located_image.extent == extent