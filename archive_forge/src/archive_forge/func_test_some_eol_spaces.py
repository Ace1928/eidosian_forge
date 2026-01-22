import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_some_eol_spaces(self):
    yaml_str = '---  \n  \na: "x"  \n   \nb: y  \n'
    d = round_trip_load(yaml_str, preserve_quotes=True)
    y = round_trip_dump(d, explicit_start=True)
    stripped = ''
    for line in yaml_str.splitlines():
        stripped += line.rstrip() + '\n'
        print(line + '$')
    assert stripped == y