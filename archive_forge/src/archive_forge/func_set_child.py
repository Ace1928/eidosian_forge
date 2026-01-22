from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def set_child(self, child):
    self.child = child