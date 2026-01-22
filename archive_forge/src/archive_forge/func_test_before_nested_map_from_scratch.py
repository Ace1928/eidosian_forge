from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_before_nested_map_from_scratch(self):
    from srsly.ruamel_yaml.comments import CommentedMap
    data = CommentedMap()
    datab = CommentedMap()
    data['a'] = 1
    data['b'] = datab
    datab['c'] = 2
    datab['d'] = 3
    data['b'].yaml_set_start_comment('Hello\nWorld\n')
    exp = '\n        a: 1\n        b:\n        # Hello\n        # World\n          c: 2\n          d: 3\n        '
    compare(data, exp.format(comment='#'))