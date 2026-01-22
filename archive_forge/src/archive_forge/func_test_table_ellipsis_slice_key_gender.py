import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_table_ellipsis_slice_key_gender(self):
    sliced = self.table['M', ...]
    if not all((el == 'M' for el in sliced.dimension_values('Gender'))):
        raise AssertionError("Table key slicing on 'Gender' failed.")