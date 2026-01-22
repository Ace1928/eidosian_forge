import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
@pytest.mark.xfail(strict=True)
def test_multispace_map_initial(self):
    round_trip('\n\n        a: 1x\n\n        b: 2x\n\n\n        c: 3x\n\n\n\n        d: 4x\n\n        ')