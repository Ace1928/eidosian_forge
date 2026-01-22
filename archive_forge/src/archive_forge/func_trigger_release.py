from kivy.clock import Clock
from kivy.config import Config
from kivy.properties import OptionProperty, ObjectProperty, \
from time import time
def trigger_release(dt):
    self._do_release()
    self.dispatch('on_release')