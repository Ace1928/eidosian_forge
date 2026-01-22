from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_double_quoted_string(self):
    inp = '\n        a: "abc"\n        '
    round_trip(inp, preserve_quotes=True)