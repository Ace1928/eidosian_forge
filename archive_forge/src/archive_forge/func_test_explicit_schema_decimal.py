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
def test_explicit_schema_decimal(self):
    rows = b'{"a": 1}\n{"a": 1.45}\n{"a": -23.456}\n{}\n'
    expected = {'a': [Decimal('1'), Decimal('1.45'), Decimal('-23.456'), None]}
    for type_factory in (pa.decimal128, pa.decimal256):
        schema = pa.schema([('a', type_factory(9, 4))])
        opts = ParseOptions(explicit_schema=schema)
        table = self.read_bytes(rows, parse_options=opts)
        assert table.schema == schema
        assert table.to_pydict() == expected