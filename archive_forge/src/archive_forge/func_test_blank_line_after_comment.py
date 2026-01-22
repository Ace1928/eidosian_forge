from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_blank_line_after_comment(self):
    round_trip('\n        # Comment with spaces after it.\n\n\n        a: 1\n        ')