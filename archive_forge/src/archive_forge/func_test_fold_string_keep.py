from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_fold_string_keep(self):
    with pytest.raises(AssertionError) as excinfo:
        inp = '\n            a: >+\n              abc\n              def\n\n            '
        round_trip(inp, intermediate=dict(a='abc def\n\n'))