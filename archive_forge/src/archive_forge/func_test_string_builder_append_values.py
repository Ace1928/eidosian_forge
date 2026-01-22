import weakref
import numpy as np
import pyarrow as pa
from pyarrow.lib import StringBuilder
def test_string_builder_append_values():
    sbuilder = StringBuilder()
    sbuilder.append_values([np.nan, None, 'text', None, 'other text'])
    assert sbuilder.null_count == 3
    arr = sbuilder.finish()
    assert arr.null_count == 3
    expected = [None, None, 'text', None, 'other text']
    assert arr.to_pylist() == expected