from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pytest  # NOQA
from .roundtrip import round_trip, round_trip_load, round_trip_dump, dedent, YAML
def test_00(self):
    yaml = YAML()
    yaml.indent = 6
    yaml.block_seq_indent = 3
    inp = '\n        a:\n           -  1\n           -  [1, 2]\n        '
    yaml.round_trip(inp)