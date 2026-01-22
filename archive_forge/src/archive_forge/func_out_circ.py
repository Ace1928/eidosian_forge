from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def out_circ(progress):
    """.. image:: images/anim_out_circ.png
        """
    p = progress - 1.0
    return sqrt(1.0 - p * p)