from kivy.app import runTouchApp
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_random_color
import timeit
import re
import random
from functools import partial
from the brougham.
def layout_real_perf(label, repeat):
    if repeat:
        repeat = int(repeat)
    else:
        return 'None'
    old_text = label._label.texture
    label._label.texture = label._label.texture_1px
    res = str(timeit.Timer(partial(label._label.render, True)).repeat(1, repeat))
    label._label.texture = old_text
    return res