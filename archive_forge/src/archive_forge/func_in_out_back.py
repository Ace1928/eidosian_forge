from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_out_back(progress):
    """.. image:: images/anim_in_out_back.png
        """
    p = progress * 2.0
    s = 1.70158 * 1.525
    if p < 1:
        return 0.5 * (p * p * ((s + 1.0) * p - s))
    p -= 2.0
    return 0.5 * (p * p * ((s + 1.0) * p + s) + 2.0)