from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_to_holomap_dask(self):
    if dd is None:
        raise SkipTest('Dask required to test .to with dask dataframe.')
    ddf = dd.from_pandas(self.df, npartitions=2)
    dds = Dataset(ddf, kdims=[Dimension('a', label='The a Column'), Dimension('b', label='The b Column'), Dimension('c', label='The c Column'), Dimension('d', label='The d Column')])
    curve_hmap = dds.to(Curve, 'a', 'b', groupby=['c'])
    for v in self.df.c.drop_duplicates():
        curve = curve_hmap.data[v,]
        self.assertEqual(curve.dataset, self.ds)
        self.assertEqual(curve.pipeline(curve.dataset), curve)