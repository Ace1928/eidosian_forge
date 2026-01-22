from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_quint(progress):
    """.. image:: images/anim_in_quint.png
        """
    return progress * progress * progress * progress * progress