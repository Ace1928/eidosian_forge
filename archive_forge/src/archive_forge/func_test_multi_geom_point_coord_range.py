from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_multi_geom_point_coord_range(self):
    geoms = [{'geometry': sgeom.Point([(0, 1)])}, {'geometry': sgeom.Point([(3, 5)])}]
    mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertEqual(mds.range('x'), (0, 3))
    self.assertEqual(mds.range('y'), (1, 5))