import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_change_key_simple_mapping_key(self):
    from srsly.ruamel_yaml.comments import CommentedKeyMap
    inp = '        {a: 1, b: 2}: hello world\n        '
    d = round_trip_load(inp, preserve_quotes=True)
    d[CommentedKeyMap([('b', 1), ('a', 2)])] = d.pop(CommentedKeyMap([('a', 1), ('b', 2)]))
    exp = dedent('        {b: 1, a: 2}: hello world\n        ')
    assert round_trip_dump(d) == exp