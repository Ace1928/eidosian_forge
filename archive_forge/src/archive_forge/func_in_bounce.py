from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_bounce(progress):
    """.. image:: images/anim_in_bounce.png
        """
    return AnimationTransition._in_bounce_internal(progress, 1.0)