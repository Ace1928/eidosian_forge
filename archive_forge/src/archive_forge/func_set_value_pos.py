from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
def set_value_pos(self, pos):
    padding = self.padding
    x = min(self.right - padding, max(pos[0], self.x + padding))
    y = min(self.top - padding, max(pos[1], self.y + padding))
    if self.orientation == 'horizontal':
        if self.width == 0:
            self.value_normalized = 0
        else:
            self.value_normalized = (x - self.x - padding) / float(self.width - 2 * padding)
    elif self.height == 0:
        self.value_normalized = 0
    else:
        self.value_normalized = (y - self.y - padding) / float(self.height - 2 * padding)