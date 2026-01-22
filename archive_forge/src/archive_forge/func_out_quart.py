from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def out_quart(progress):
    """.. image:: images/anim_out_quart.png
        """
    p = progress - 1.0
    return -1.0 * (p * p * p * p - 1.0)