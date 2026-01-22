from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_map_set_comment_before_and_after_non_first_key_02(self):
    data = load('\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        test1:\n          test2:\n            test3: 3\n        ')
    data.yaml_set_comment_before_after_key('test1', 'xyz\n\nbefore test1 (top level)', after='\nbefore test2', after_indent=4)
    data['test1']['test2'].yaml_set_start_comment('after test2', indent=4)
    exp = '\n        xyz:\n          a: 1    # comment 1\n          b: 2\n\n        # xyz\n\n        # before test1 (top level)\n        test1:\n\n            # before test2\n          test2:\n            # after test2\n            test3: 3\n        '
    compare(data, exp)