from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_concat_tables_invalid_option():
    t = pa.Table.from_arrays([list(range(10))], names=('a',))
    with pytest.raises(ValueError, match='Invalid promote options: invalid'):
        pa.concat_tables([t, t], promote_options='invalid')