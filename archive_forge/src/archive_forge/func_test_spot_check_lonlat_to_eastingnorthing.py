import numpy as np
import pandas as pd
from holoviews import Tiles
from holoviews.element.comparison import ComparisonTestCase
def test_spot_check_lonlat_to_eastingnorthing(self):
    easting, northing = Tiles.lon_lat_to_easting_northing(0, 0)
    self.assertAlmostEqual(easting, 0)
    self.assertAlmostEqual(northing, 0)
    easting, northing = Tiles.lon_lat_to_easting_northing(20, 10)
    self.assertAlmostEqual(easting, 2226389.82, places=2)
    self.assertAlmostEqual(northing, 1118889.97, places=2)
    easting, northing = Tiles.lon_lat_to_easting_northing(-33, -18)
    self.assertAlmostEqual(easting, -3673543.2, places=2)
    self.assertAlmostEqual(northing, -2037548.54, places=2)
    easting, northing = Tiles.lon_lat_to_easting_northing(85, -75)
    self.assertAlmostEqual(easting, 9462156.72, places=2)
    self.assertAlmostEqual(northing, -12932243.11, places=2)
    easting, northing = Tiles.lon_lat_to_easting_northing(180, 85)
    self.assertAlmostEqual(easting, 20037508.34, places=2)
    self.assertAlmostEqual(northing, 19971868.88, places=2)