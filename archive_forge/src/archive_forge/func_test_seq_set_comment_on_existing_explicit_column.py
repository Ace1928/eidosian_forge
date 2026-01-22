from __future__ import print_function
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_seq_set_comment_on_existing_explicit_column(self):
    data = load('\n        - a   # comment 1\n        - b\n        - c\n        ')
    data.yaml_add_eol_comment('comment 2', key=1, column=6)
    exp = '\n        - a   # comment 1\n        - b   # comment 2\n        - c\n        '
    compare(data, exp)