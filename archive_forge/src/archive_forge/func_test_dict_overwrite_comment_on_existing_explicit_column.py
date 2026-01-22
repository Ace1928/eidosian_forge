from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_dict_overwrite_comment_on_existing_explicit_column(self):
    data = load('\n        a: 1   # comment 1\n        b: 2\n        c: 3\n        d: 4\n        e: 5\n        ')
    data.yaml_add_eol_comment('comment 2', key='a', column=7)
    exp = '\n        a: 1   # comment 2\n        b: 2\n        c: 3\n        d: 4\n        e: 5\n        '
    compare(data, exp)