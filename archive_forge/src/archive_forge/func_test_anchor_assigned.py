import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_anchor_assigned(self):
    from srsly.ruamel_yaml.comments import CommentedMap
    data = load('\n        a: &id002\n          b: 1\n          c: 2\n        d: *id002\n        e: &etemplate\n          b: 1\n          c: 2\n        f: *etemplate\n        ')
    d = data['d']
    assert isinstance(d, CommentedMap)
    assert d.yaml_anchor() is None
    e = data['e']
    assert isinstance(e, CommentedMap)
    assert e.yaml_anchor().value == 'etemplate'
    assert e.yaml_anchor().always_dump is False