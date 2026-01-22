from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_elastic(progress):
    """.. image:: images/anim_in_elastic.png
        """
    p = 0.3
    s = p / 4.0
    q = progress
    if q == 1:
        return 1.0
    q -= 1.0
    return -(pow(2, 10 * q) * sin((q - s) * (2 * pi) / p))