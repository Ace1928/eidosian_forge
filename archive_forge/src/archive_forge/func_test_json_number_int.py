from __future__ import print_function
import pytest  # NOQA
import json
def test_json_number_int(self):
    for x in (y.split('#')[0].strip() for y in '\n        42\n        '.splitlines()):
        if not x:
            continue
        res = load(x, int)
        assert isinstance(res, int)