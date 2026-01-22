import pytest
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_set_simple_mapping_key(self):
    from srsly.ruamel_yaml.comments import CommentedKeyMap
    d = {CommentedKeyMap([('a', 1), ('b', 2)]): 'hello world'}
    exp = dedent('        {a: 1, b: 2}: hello world\n        ')
    assert round_trip_dump(d) == exp