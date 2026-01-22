import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_template_matcher(self):
    """test if id matches the anchor template"""
    from srsly.ruamel_yaml.serializer import templated_id
    assert templated_id(u'id001')
    assert templated_id(u'id999')
    assert templated_id(u'id1000')
    assert templated_id(u'id0001')
    assert templated_id(u'id0000')
    assert not templated_id(u'id02')
    assert not templated_id(u'id000')
    assert not templated_id(u'x000')