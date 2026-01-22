from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def find_touch(self, win, x, y):
    factor = 10.0 / win.system_size[0]
    for touch in self.touches.values():
        if abs(x - touch.sx) < factor and abs(y - touch.sy) < factor:
            return touch
    return None