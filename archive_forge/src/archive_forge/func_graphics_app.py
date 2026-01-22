from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def graphics_app():
    from kivy.app import App
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, Rectangle

    class TestApp(UnitKivyApp, App):

        def build(self):
            widget = Widget()
            with widget.canvas:
                Color(1, 0, 0, 1)
                Rectangle(pos=(0, 0), size=(100, 100))
                Color(0, 1, 0, 1)
                Rectangle(pos=(100, 0), size=(100, 100))
            return widget
    return TestApp()