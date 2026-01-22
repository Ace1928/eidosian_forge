import random
import time
import math
import os
from collections import deque
from kivy.tests import UnitTestTouch
def family_up(self, **kwargs_filter):
    self.match(**kwargs_filter)
    check = self.check_widget
    base_widget = self.base_widget
    already_checked_base = None
    while base_widget is not None:
        fifo = deque([base_widget])
        while fifo:
            widget = fifo.popleft()
            if widget is already_checked_base:
                continue
            if check(widget):
                return WidgetResolver(base_widget=widget)
            fifo.extend(widget.children)
        already_checked_base = base_widget
        new_base_widget = base_widget.parent
        if new_base_widget is base_widget:
            break
        base_widget = new_base_widget
    self.not_found('family_up')