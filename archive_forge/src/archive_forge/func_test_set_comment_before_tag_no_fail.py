import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_set_comment_before_tag_no_fail(self):
    inp = '\n        # the beginning\n        !!set\n        # or this one?\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        '
    assert round_trip_dump(round_trip_load(inp)) == dedent('\n        !!set\n        # or this one?\n        ? a\n        # next one is B (lowercase)\n        ? b  #  You see? Promised you.\n        ? c\n        # this is the end\n        ')