import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_simple_mapping_key(self):
    inp = '        {a: 1, b: 2}: hello world\n        '
    round_trip(inp, preserve_quotes=True, dump_data=False)