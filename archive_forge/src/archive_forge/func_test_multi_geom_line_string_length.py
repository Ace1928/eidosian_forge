from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_multi_geom_line_string_length(self):
    geoms = [{'geometry': sgeom.LineString([(0, 0), (3, 3), (6, 0)])}, {'geometry': sgeom.LineString([(3, 3), (9, 3), (6, 0)])}]
    mds = Dataset(geoms, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertEqual(len(mds), 2)