import random
import time
import math
import os
from collections import deque
from kivy.tests import UnitTestTouch
def resolve_widget(self, base_widget=None):
    if base_widget is None:
        from kivy.core.window import Window
        base_widget = Window
    return WidgetResolver(base_widget=base_widget)