from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
@pytest.mark.xfail(raises=ParseError, reason='Bad XML returned from the URL')
def test_fetch_geometries(self):
    source = ogc.WFSGeometrySource(self.URI, self.typename)
    extent = (-99012, 1523166, -6740315, -4589165)
    geoms = source.fetch_geometries(self.native_projection, extent)
    assert len(geoms) == 23