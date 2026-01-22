from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_reindex_dataset(self):
    ds_ab = self.ds.reindex(kdims=['a'], vdims=['b'])
    ds2_ab = self.ds2.reindex(kdims=['a'], vdims=['b'])
    self.assertNotEqual(ds_ab, ds2_ab)
    self.assertEqual(ds_ab.dataset, self.ds)
    ops = ds_ab.pipeline.operations
    self.assertEqual(len(ops), 2)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertEqual(ops[1].method_name, 'reindex')
    self.assertEqual(ops[1].args, [])
    self.assertEqual(ops[1].kwargs, dict(kdims=['a'], vdims=['b']))
    self.assertEqual(ds_ab.pipeline(ds_ab.dataset), ds_ab)
    self.assertEqual(ds_ab.pipeline(self.ds2), ds2_ab)