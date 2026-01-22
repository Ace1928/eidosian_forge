from collections import OrderedDict
from decimal import Decimal
import io
import itertools
import json
import string
import unittest
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.json import read_json, ReadOptions, ParseOptions
def make_random_json(num_cols=2, num_rows=10, linesep='\r\n'):
    arr = np.random.RandomState(42).randint(0, 1000, size=(num_cols, num_rows))
    col_names = list(itertools.islice(generate_col_names(), num_cols))
    lines = []
    for row in arr.T:
        json_obj = OrderedDict([(k, int(v)) for k, v in zip(col_names, row)])
        lines.append(json.dumps(json_obj))
    data = linesep.join(lines).encode()
    columns = [pa.array(col, type=pa.int64()) for col in arr]
    expected = pa.Table.from_arrays(columns, col_names)
    return (data, expected)