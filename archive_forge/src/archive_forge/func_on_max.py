from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
def on_max(self, *largs):
    self.value = min(self.max, max(self.min, self.value))