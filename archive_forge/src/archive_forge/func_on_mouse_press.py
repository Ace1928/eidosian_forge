from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def on_mouse_press(self, win, x, y, button, modifiers):
    if self.test_activity():
        return
    nx, ny = win.to_normalized_pos(x, y)
    ny = 1.0 - ny
    found_touch = self.find_touch(win, nx, ny)
    if found_touch:
        self.current_drag = found_touch
    else:
        is_double_tap = 'shift' in modifiers
        do_graphics = not self.disable_multitouch and (button != 'left' or 'ctrl' in modifiers)
        touch = self.create_touch(win, nx, ny, is_double_tap, do_graphics, button)
        if 'alt' in modifiers:
            self.alt_touch = touch
            self.current_drag = None