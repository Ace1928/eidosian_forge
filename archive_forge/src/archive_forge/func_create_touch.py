from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def create_touch(self, win, nx, ny, is_double_tap, do_graphics, button):
    event_id = self.create_event_id()
    args = [nx, ny, button]
    if do_graphics:
        args += [not self.multitouch_on_demand]
    self.current_drag = touch = MouseMotionEvent(self.device, event_id, args, is_touch=True, type_id='touch')
    touch.is_double_tap = is_double_tap
    self.touches[event_id] = touch
    if do_graphics:
        create_flag = not self.disable_multitouch and (not self.multitouch_on_demand)
        touch.update_graphics(win, create_flag)
    self.waiting_event.append(('begin', touch))
    return touch