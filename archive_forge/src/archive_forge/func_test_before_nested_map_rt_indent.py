from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_nested_map_rt_indent(self):
    data = load('\n        a: 1\n        b:\n          c: 2\n          d: 3\n        ')
    data['b'].yaml_set_start_comment('Hello\nWorld\n', indent=2)
    exp = '\n        a: 1\n        b:\n          # Hello\n          # World\n          c: 2\n          d: 3\n        '
    compare(data, exp.format(comment='#'))
    print(data['b'].ca)