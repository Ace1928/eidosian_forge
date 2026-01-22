import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_add_well_known_multiheader(self, d):
    d.add('COOKIE', 'asdf')
    assert d.getlist('cookie') == ['foo', 'bar', 'asdf']
    assert d['cookie'] == 'foo, bar, asdf'