from kivy.animation import Animation
from kivy.properties import (
from kivy.uix.anchorlayout import AnchorLayout
def on__anim_alpha(self, _instance, value):
    """ animation progress callback. """
    if value == 0 and self._is_open:
        self._real_remove_widget()