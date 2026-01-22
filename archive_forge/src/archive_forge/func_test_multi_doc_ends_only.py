import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load_all
def test_multi_doc_ends_only(self):
    inp = '        - a\n        ...\n        - b\n        ...\n        '
    docs = list(round_trip_load_all(inp, version=(1, 2)))
    assert docs == [['a'], ['b']]