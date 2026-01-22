from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_concat_tables_with_promotion_error():
    t1 = pa.Table.from_arrays([pa.array([1, 2], type=pa.int64())], ['f'])
    t2 = pa.Table.from_arrays([pa.array([1, 2], type=pa.float32())], ['f'])
    with pytest.raises(pa.ArrowTypeError, match='Unable to merge:'):
        pa.concat_tables([t1, t2], promote_options='default')