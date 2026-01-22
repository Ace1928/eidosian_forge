import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_change_value_simple_mapping_key(self):
    from srsly.ruamel_yaml.comments import CommentedKeyMap
    inp = '        {a: 1, b: 2}: hello world\n        '
    d = round_trip_load(inp, preserve_quotes=True)
    d = {CommentedKeyMap([('a', 1), ('b', 2)]): 'goodbye'}
    exp = dedent('        {a: 1, b: 2}: goodbye\n        ')
    assert round_trip_dump(d) == exp