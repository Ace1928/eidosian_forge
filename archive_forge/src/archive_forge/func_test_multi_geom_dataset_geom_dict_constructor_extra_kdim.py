from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_multi_geom_dataset_geom_dict_constructor_extra_kdim(self):
    geoms = [{'geometry': sgeom.Polygon([(0, 0), (3, 3), (6, 0)]), 'z': 1}]
    Dataset(geoms, kdims=['x', 'y', 'z'], datatype=[self.datatype])