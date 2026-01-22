from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def clear_graphics(self, win):
    de = self.ud.pop('_drawelement', None)
    if de is not None:
        win.canvas.after.remove(de[0])
        win.canvas.after.remove(de[1])