from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_concat_tables_permissive():
    t1 = pa.Table.from_arrays([list(range(10))], names=('a',))
    t2 = pa.Table.from_arrays([list(('a', 'b', 'c'))], names=('a',))
    with pytest.raises(pa.ArrowTypeError, match='Unable to merge: Field a has incompatible types: int64 vs string'):
        _ = pa.concat_tables([t1, t2], promote_options='permissive')