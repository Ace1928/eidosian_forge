from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def rescale_parent_proportion(self, *args):
    if not self.parent:
        return
    if self.rescale_with_parent:
        parent_proportion = self._parent_proportion
        if self.sizable_from in ('top', 'bottom'):
            new_height = parent_proportion * self.parent.height
            self.height = max(self.min_size, min(new_height, self.max_size))
        else:
            new_width = parent_proportion * self.parent.width
            self.width = max(self.min_size, min(new_width, self.max_size))