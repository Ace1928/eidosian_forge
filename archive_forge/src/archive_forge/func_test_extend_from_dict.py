import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_extend_from_dict(self, d):
    d.extend(dict(cookie='asdf'), b='100')
    assert d['cookie'] == 'foo, bar, asdf'
    assert d['b'] == '100'
    d.add('cookie', 'with, comma')
    assert d.getlist('cookie') == ['foo', 'bar', 'asdf', 'with, comma']