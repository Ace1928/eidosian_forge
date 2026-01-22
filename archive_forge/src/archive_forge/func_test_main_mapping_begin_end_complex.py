import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_main_mapping_begin_end_complex(self):
    round_trip('\n        # C start a\n        # C start b\n        abc: 1\n        ghi: 2\n        klm:\n          3a: alpha\n          3b: beta   # it is all greek to me\n        # C end a\n        # C end b\n        ')