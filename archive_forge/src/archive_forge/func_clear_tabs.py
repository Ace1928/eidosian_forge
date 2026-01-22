from functools import partial
from kivy.clock import Clock
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty, OptionProperty, \
def clear_tabs(self, *l):
    self_tabs = self._tab_strip
    self_tabs.clear_widgets()
    if self.do_default_tab:
        self_default_tab = self._default_tab
        self_tabs.add_widget(self_default_tab)
        self_tabs.width = self_default_tab.width
    self._reposition_tabs()