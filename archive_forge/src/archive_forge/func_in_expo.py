from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_expo(progress):
    """.. image:: images/anim_in_expo.png
        """
    if progress == 0:
        return 0.0
    return pow(2, 10 * (progress - 1.0))