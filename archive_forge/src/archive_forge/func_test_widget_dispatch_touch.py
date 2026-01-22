import pytest
from string import ascii_letters
from random import randint
import gc
import sys
@pytest.mark.parametrize('n', [1, 10, 100, 1000])
def test_widget_dispatch_touch(kivy_benchmark, n):
    from kivy.tests.common import UnitTestTouch
    from kivy.uix.widget import Widget
    root = Widget()
    for x in range(10):
        parent = Widget()
        for y in range(n):
            parent.add_widget(Widget())
        root.add_widget(parent)
    touch = UnitTestTouch(10, 10)

    def dispatch():
        root.dispatch('on_touch_down', touch)
        root.dispatch('on_touch_move', touch)
        root.dispatch('on_touch_up', touch)
    kivy_benchmark(dispatch)