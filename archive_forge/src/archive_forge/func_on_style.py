from pygments import highlight
from pygments import lexers
from pygments import styles
from pygments.formatters import BBCodeFormatter
from kivy.uix.textinput import TextInput
from kivy.core.text.markup import MarkupLabel as Label
from kivy.cache import Cache
from kivy.properties import ObjectProperty, OptionProperty
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.behaviors import CodeNavigationBehavior
def on_style(self, *args):
    self.formatter = BBCodeFormatter(style=self.style)
    self._trigger_update_graphics()