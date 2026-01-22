from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def out_bounce(progress):
    """.. image:: images/anim_out_bounce.png
        """
    return AnimationTransition._out_bounce_internal(progress, 1.0)