import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_anchor_id_renumber(self):
    from srsly.ruamel_yaml.serializer import Serializer
    assert Serializer.ANCHOR_TEMPLATE == 'id%03d'
    data = load('\n        a: &id002\n          b: 1\n          c: 2\n        d: *id002\n        ')
    compare(data, '\n        a: &id001\n          b: 1\n          c: 2\n        d: *id001\n        ')