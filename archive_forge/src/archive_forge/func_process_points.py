from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.tests.common import UTMotionEvent
from time import sleep
from itertools import count
def process_points(self, scroll, points):
    win = EventLoop.window
    dt = 0.02
    for point in points:
        if DEBUG:
            print('point:', point, scroll.scroll_x, scroll.scroll_y)
            Clock.schedule_once(lambda *dt: sleep(0.5), 0)
            self.render(scroll)
        x, y, nx, ny, pos_x, pos_y, border_check = point
        scroll.bar_pos = (pos_x, pos_y)
        touch = UTMotionEvent('unittest', next(touch_id), {'x': x / float(win.width), 'y': y / float(win.height)})
        self.assertAlmostEqual(scroll.scroll_x, 0.0, delta=dt)
        self.assertAlmostEqual(scroll.scroll_y, 1.0, delta=dt)
        if border_check:
            EventLoop.post_dispatch_input('begin', touch)
            touch.move({'x': nx / float(win.width), 'y': ny / float(win.height)})
            EventLoop.post_dispatch_input('update', touch)
            EventLoop.post_dispatch_input('end', touch)
            self.assertAlmostEqual(scroll.scroll_x, 0.0, delta=dt)
            self.assertAlmostEqual(scroll.scroll_y, 1.0, delta=dt)
            return
        EventLoop.post_dispatch_input('begin', touch)
        touch.move({'x': nx / float(win.width), 'y': ny / float(win.height)})
        EventLoop.post_dispatch_input('update', touch)
        EventLoop.post_dispatch_input('end', touch)
        if DEBUG:
            print(scroll.scroll_x, scroll.scroll_y)
            Clock.schedule_once(lambda *dt: sleep(0.5), 0)
            self.render(scroll)
        self.assertAlmostEqual(scroll.scroll_x, 0.0 if x == nx else 1.0, delta=dt)
        self.assertAlmostEqual(scroll.scroll_y, 1.0 if y == ny else 0.0, delta=dt)
        scroll.scroll_x = 0.0
        scroll.scroll_y = 1.0