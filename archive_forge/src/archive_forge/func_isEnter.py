import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def isEnter(self):
    """Returns True if the mouse has just entered the item's shape"""
    return self.enter