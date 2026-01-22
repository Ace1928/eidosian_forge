import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
@property
def dpos(self):
    """Return delta between last position and current position, in the
        screen coordinate system (self.dx, self.dy)."""
    return (self.dx, self.dy)