from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.tests.common import UTMotionEvent
from time import sleep
from itertools import count
def test_smooth_scroll_end(self):
    EventLoop.ensure_window()
    win = EventLoop.window
    grid = _TestGrid()
    scroll = ScrollView(smooth_scroll_end=10)
    assert scroll.smooth_scroll_end == 10
    scroll.add_widget(grid)
    while win.children:
        win.remove_widget(win.children[0])
    win.add_widget(scroll)
    EventLoop.idle()
    e = scroll.effect_y
    assert e.velocity == 0
    touch = UTMotionEvent('unittest', next(touch_id), {'x': scroll.center_x / float(win.width), 'y': scroll.center_y / float(win.height)})
    touch.profile.append('button')
    touch.button = 'scrollup'
    EventLoop.post_dispatch_input('begin', touch)
    assert e.velocity == 10 * scroll.scroll_wheel_distance
    EventLoop.idle()
    assert 0 < e.velocity < 10 * scroll.scroll_wheel_distance
    EventLoop.post_dispatch_input('end', touch)
    EventLoop.idle()
    assert 0 < e.velocity < 10 * scroll.scroll_wheel_distance
    while e.velocity:
        EventLoop.idle()
    touch = UTMotionEvent('unittest', next(touch_id), {'x': scroll.center_x / float(win.width), 'y': scroll.center_y / float(win.height)})
    touch.profile.append('button')
    touch.button = 'scrolldown'
    EventLoop.post_dispatch_input('begin', touch)
    assert e.velocity == -10 * scroll.scroll_wheel_distance
    EventLoop.idle()
    assert 0 > e.velocity > -10 * scroll.scroll_wheel_distance
    EventLoop.post_dispatch_input('end', touch)
    EventLoop.idle()
    assert 0 > e.velocity > -10 * scroll.scroll_wheel_distance