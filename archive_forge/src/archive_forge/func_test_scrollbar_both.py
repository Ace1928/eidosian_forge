from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.tests.common import UTMotionEvent
from time import sleep
from itertools import count
def test_scrollbar_both(self):
    EventLoop.ensure_window()
    win = EventLoop.window
    grid = _TestGrid()
    scroll = _TestScrollbarBoth()
    scroll.add_widget(grid)
    win.add_widget(scroll)
    EventLoop.idle()
    left, right = scroll.to_window(scroll.x, scroll.right)
    bottom, top = scroll.to_window(scroll.y, scroll.top)
    points = [[left, bottom, right, bottom, 'bottom', 'right', False], [left, top, right, top, 'top', 'right', False], [right, top, right, bottom, 'bottom', 'right', False], [left, top, left, bottom, 'bottom', 'left', False]]
    self.process_points(scroll, points)
    self.render(scroll)