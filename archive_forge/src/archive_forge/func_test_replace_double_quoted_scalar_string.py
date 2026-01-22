from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
def test_replace_double_quoted_scalar_string(self):
    import srsly
    s = dedent('        foo: "foo foo bar foo"\n        ')
    data = round_trip_load(s, preserve_quotes=True)
    so = data['foo'].replace('foo', 'bar', 2)
    assert isinstance(so, srsly.ruamel_yaml.scalarstring.DoubleQuotedScalarString)
    assert so == 'bar bar bar foo'