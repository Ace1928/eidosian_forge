from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def button_app():
    from kivy.app import App
    from kivy.uix.togglebutton import ToggleButton

    class TestApp(UnitKivyApp, App):

        def build(self):
            return ToggleButton(text='Hello, World!')
    return TestApp()