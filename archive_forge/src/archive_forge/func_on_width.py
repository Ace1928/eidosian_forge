from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.config import Config
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty, \
from kivy.metrics import sp
from kivy.lang import Builder
from functools import partial
def on_width(self, width, *args):
    total_width = 0
    for child in self._list_action_items:
        total_width += child.pack_width
    for group in self._list_action_group:
        for child in group.list_action_item:
            total_width += child.pack_width
    if total_width <= self.width:
        if self._state != 'all':
            self._layout_all()
        return
    total_width = 0
    for child in self._list_action_items:
        total_width += child.pack_width
    for group in self._list_action_group:
        total_width += group.pack_width
    if total_width < self.width:
        if self._state != 'group':
            self._layout_group()
        return
    self._layout_random()