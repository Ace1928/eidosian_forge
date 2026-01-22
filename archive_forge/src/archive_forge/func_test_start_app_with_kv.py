from os import name
import os.path
from math import isclose
from textwrap import dedent
from kivy.app import App
from kivy.clock import Clock
from kivy import lang
from kivy.tests import GraphicUnitTest, async_run, UnitKivyApp
def test_start_app_with_kv(self):

    class TestKvApp(App):
        pass
    lang._delayed_start = None
    a = TestKvApp()
    Clock.schedule_once(a.stop, 0.1)
    a.run()