import random
import time
import math
import os
from collections import deque
from kivy.tests import UnitTestTouch
def check_widget(self, widget):
    if not all((func(widget) for func in self._funcs_filter)):
        return False
    for attr, val in self._kwargs_filter.items():
        if getattr(widget, attr, _unique_value) != val:
            return False
    return True