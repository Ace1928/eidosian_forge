from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_multi_dict_groupby(self):
    geoms = [{'geometry': sgeom.Polygon([(2, 0), (1, 2), (0, 0)]), 'z': 1}, {'geometry': sgeom.Polygon([(3, 3), (3, 3), (6, 0)]), 'z': 2}]
    mds = Dataset(geoms, kdims=['x', 'y', 'z'], datatype=[self.datatype])
    for i, (k, ds) in enumerate(mds.groupby('z').items()):
        self.assertEqual(k, geoms[i]['z'])
        self.assertEqual(ds.clone(vdims=[]), Dataset([geoms[i]], kdims=['x', 'y']))