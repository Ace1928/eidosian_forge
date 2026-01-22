from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_df_dataset(self):
    if not pd:
        raise SkipTest('Pandas not available')
    dfs = [pd.DataFrame(np.column_stack([np.arange(i, i + 2), np.arange(i, i + 2)]), columns=['x', 'y']) for i in range(2)]
    mds = Path(dfs, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    for i, ds in enumerate(mds.split(datatype='dataframe')):
        ds['x'] = ds.x.astype(int)
        ds['y'] = ds.y.astype(int)
        self.assertEqual(ds, dfs[i])