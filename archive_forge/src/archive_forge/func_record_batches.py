import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
@st.composite
def record_batches(draw, type, rows=None, max_fields=None):
    if isinstance(rows, st.SearchStrategy):
        rows = draw(rows)
    elif rows is None:
        rows = draw(_default_array_sizes)
    elif not isinstance(rows, int):
        raise TypeError('Rows must be an integer')
    schema = draw(schemas(type, max_fields=max_fields))
    children = [draw(arrays(field.type, size=rows)) for field in schema]
    return pa.RecordBatch.from_arrays(children, schema=schema)