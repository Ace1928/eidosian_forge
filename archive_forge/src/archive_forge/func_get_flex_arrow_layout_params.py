from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import OptionProperty
from kivy.properties import ListProperty
from kivy.properties import BooleanProperty
from kivy.properties import ColorProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.base import EventLoop
from kivy.metrics import dp
def get_flex_arrow_layout_params(self):
    pos = self.flex_arrow_pos
    if pos is None:
        return None
    x, y = pos
    if not (0 <= x <= self.width and 0 <= y <= self.height):
        return None
    base_layouts_map = [('bottom_mid', y), ('top_mid', self.height - y), ('left_mid', x), ('right_mid', self.width - x)]
    base_layout_key = min(base_layouts_map, key=lambda val: val[1])[0]
    arrow_layout = list(Bubble.ARROW_LAYOUTS[base_layout_key])
    arrow_width = self._arrow_image.width

    def calc_x0(x, length):
        return x * (length - arrow_width) / (length * length)
    if base_layout_key == 'bottom_mid':
        arrow_layout[-1] = {'top': 1.0, 'x': calc_x0(x, self.width)}
    elif base_layout_key == 'top_mid':
        arrow_layout[-1] = {'bottom': 0.0, 'x': calc_x0(x, self.width)}
    elif base_layout_key == 'left_mid':
        arrow_layout[-1] = {'right': 1.0, 'y': calc_x0(y, self.height)}
    elif base_layout_key == 'right_mid':
        arrow_layout[-1] = {'left': 0.0, 'y': calc_x0(y, self.height)}
    return arrow_layout