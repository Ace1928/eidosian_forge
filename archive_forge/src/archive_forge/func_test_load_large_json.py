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
def test_load_large_json(self):
    data, expected = make_random_json(num_cols=2, num_rows=100100)
    read_options = ReadOptions(block_size=1024 * 1024 * 10)
    table = self.read_bytes(data, read_options=read_options)
    assert table.num_rows == 100100
    assert expected.num_rows == 100100