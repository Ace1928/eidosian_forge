from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_can_combine_chunks_with_no_chunks():
    assert pa.chunked_array([], type=pa.bool_()).combine_chunks() == pa.array([], type=pa.bool_())
    assert pa.chunked_array([pa.array([], type=pa.bool_())], type=pa.bool_()).combine_chunks() == pa.array([], type=pa.bool_())