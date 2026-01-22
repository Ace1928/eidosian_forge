import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_len_items_delete(self):
    from srsly.ruamel_yaml import safe_load
    from srsly.ruamel_yaml.compat import PY3
    d = safe_load(self.yaml_str)
    data = round_trip_load(self.yaml_str)
    x = data[2].items()
    print('d2 items', d[2].items(), len(d[2].items()), x, len(x))
    ref = len(d[2].items())
    print('ref', ref)
    assert len(x) == ref
    del data[2]['m']
    if PY3:
        ref -= 1
    assert len(x) == ref
    del data[2]['d']
    if PY3:
        ref -= 1
    assert len(x) == ref
    del data[2]['a']
    if PY3:
        ref -= 1
    assert len(x) == ref