from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_top_seq_rt(self):
    data = load('\n        - a\n        - b\n        ')
    data.yaml_set_start_comment('Hello\nWorld\n')
    print(round_trip_dump(data))
    exp = '\n        # Hello\n        # World\n        - a\n        - b\n        '
    compare(data, exp)