from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_nested_genlets(self):
    seen = []
    for ii in ax(5):
        seen.append(ii)