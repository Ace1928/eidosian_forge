import pytest
from string import ascii_letters
from random import randint
import gc
import sys
def test_widget_creation(kivy_benchmark):
    from kivy.uix.widget import Widget
    w = Widget()
    kivy_benchmark(Widget)