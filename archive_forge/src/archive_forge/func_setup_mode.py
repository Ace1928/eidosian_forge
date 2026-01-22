from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
def setup_mode(self, *largs):
    """Call this method when you want to readjust the keyboard according to
        options: :attr:`docked` or not, with attached :attr:`target` or not:

        * If :attr:`docked` is True, it will call :meth:`setup_mode_dock`
        * If :attr:`docked` is False, it will call :meth:`setup_mode_free`

        Feel free to overload these methods to create new
        positioning behavior.
        """
    if self.docked:
        self.setup_mode_dock()
    else:
        self.setup_mode_free()