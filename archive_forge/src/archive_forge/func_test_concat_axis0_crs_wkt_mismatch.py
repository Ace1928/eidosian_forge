import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
def test_concat_axis0_crs_wkt_mismatch(self):
    wkt_template = 'GEOGCRS["WGS 84",\n        ENSEMBLE["World Geodetic System 1984 ensemble",\n        MEMBER["World Geodetic System 1984 (Transit)"],\n        MEMBER["World Geodetic System 1984 (G730)"],\n        MEMBER["World Geodetic System 1984 (G873)"],\n        MEMBER["World Geodetic System 1984 (G1150)"],\n        MEMBER["World Geodetic System 1984 (G1674)"],\n        MEMBER["World Geodetic System 1984 (G1762)"],\n        MEMBER["World Geodetic System 1984 (G2139)"],\n        ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],\n        ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,\n        ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],\n        AXIS["geodetic latitude (Lat)",north,ORDER[1],\n        ANGLEUNIT["degree",0.0174532925199433]],\n        AXIS["geodetic longitude (Lon)",east,ORDER[2],\n        ANGLEUNIT["degree",0.0174532925199433]],\n        USAGE[SCOPE["Horizontal component of 3D system."],\n        AREA["World.{}"],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
    wkt_v1 = wkt_template.format('')
    wkt_v2 = wkt_template.format(' ')
    crs1 = pyproj.CRS.from_wkt(wkt_v1)
    crs2 = pyproj.CRS.from_wkt(wkt_v2)
    assert len({crs1, crs2}) == 2
    assert crs1 == crs2
    expected = pd.concat([self.gdf, self.gdf]).set_crs(crs1)
    res = pd.concat([self.gdf.set_crs(crs1), self.gdf.set_crs(crs2)])
    assert_geodataframe_equal(expected, res)