from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def strip_up(self, instance, touch):
    if touch.grab_current is not instance:
        return
    if touch.is_double_tap:
        max_size = self.max_size
        min_size = self.min_size
        sz_frm = self.sizable_from[0]
        s = self.size
        if sz_frm in ('t', 'b'):
            if self.size_hint_y:
                self.size_hint_y = None
            if s[1] - min_size <= max_size - s[1]:
                self.height = max_size
            else:
                self.height = min_size
        else:
            if self.size_hint_x:
                self.size_hint_x = None
            if s[0] - min_size <= max_size - s[0]:
                self.width = max_size
            else:
                self.width = min_size
    touch.ungrab(instance)
    self.dispatch('on_release')