import os
from os.path import sep
def should_use_mouse(self):
    return self.use_mouse or not any((p for p in EventLoop.input_providers if isinstance(p, MouseMotionEventProvider)))