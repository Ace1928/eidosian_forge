import copy
import pytest  # NOQA
from .roundtrip import dedent, round_trip_load, round_trip_dump
def test_preserve_flow_style_simple(self):
    x = dedent('        {foo: bar, baz: quux}\n        ')
    data = round_trip_load(x)
    data_copy = copy.deepcopy(data)
    y = round_trip_dump(data_copy)
    print('x [{}]'.format(x))
    print('y [{}]'.format(y))
    assert y == x
    assert data.fa.flow_style() == data_copy.fa.flow_style()