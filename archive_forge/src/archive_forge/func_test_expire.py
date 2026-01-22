import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_expire(self):
    d = Container(5)
    for i in xrange(5):
        d[i] = str(i)
    for i in xrange(5):
        d.get(0)
    d[5] = '5'
    assert list(d.keys()) == [2, 3, 4, 0, 5]