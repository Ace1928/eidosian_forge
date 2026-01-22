from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_top_map_replace(self):
    data = load('\n        # abc\n        # def\n        a: 1 # 1\n        b: 2\n        ')
    data.yaml_set_start_comment('Hello\nWorld\n')
    exp = '\n        # Hello\n        # World\n        a: 1 # 1\n        b: 2\n        '
    compare(data, exp.format(comment='#'))