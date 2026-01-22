import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_maxsize(self):
    d = Container(5)
    for i in xrange(5):
        d[i] = str(i)
    assert len(d) == 5
    for i in xrange(5):
        assert d[i] == str(i)
    d[i + 1] = str(i + 1)
    assert len(d) == 5
    assert 0 not in d
    assert i + 1 in d