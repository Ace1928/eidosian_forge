from itertools import product
from kivy.tests import GraphicUnitTest
from kivy.logger import LoggerHistory
def test_window_opacity_property(self):
    if self.check_opacity_support():
        opacity = self.get_new_opacity_value()
        self.Window.opacity = opacity
        self.assertEqual(self.Window.opacity, opacity)