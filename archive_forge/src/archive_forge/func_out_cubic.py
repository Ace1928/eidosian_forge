from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def out_cubic(progress):
    """.. image:: images/anim_out_cubic.png
        """
    p = progress - 1.0
    return p * p * p + 1.0