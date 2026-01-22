from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def stop_property(self, widget, prop):
    self.anim1.stop_property(widget, prop)
    self.anim2.stop_property(widget, prop)
    if not self.anim1.have_properties_to_animate(widget) and (not self.anim2.have_properties_to_animate(widget)):
        self.stop(widget)