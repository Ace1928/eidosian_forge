import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
def test_create_from_dict(self):
    h = HTTPHeaderDict(dict(ab=1, cd=2, ef=3, gh=4))
    assert len(h) == 4
    assert 'ab' in h