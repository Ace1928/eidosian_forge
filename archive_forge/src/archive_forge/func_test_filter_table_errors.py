import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType
def test_filter_table_errors():
    t = pa.table({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
    with pytest.raises(pa.ArrowTypeError):
        _filter_table(t, pc.divide(pc.field('a'), pc.scalar(2)))
    with pytest.raises(pa.ArrowInvalid):
        _filter_table(t, pc.field('Z') <= pc.scalar(2))