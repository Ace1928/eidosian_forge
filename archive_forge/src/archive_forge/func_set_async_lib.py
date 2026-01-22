import random
import time
import math
import os
from collections import deque
from kivy.tests import UnitTestTouch
def set_async_lib(self, async_lib):
    from kivy.clock import Clock
    if async_lib is not None:
        Clock.init_async_lib(async_lib)
    self.async_sleep = Clock._async_lib.sleep