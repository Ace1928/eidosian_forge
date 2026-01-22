from json import load
from os.path import exists
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
from kivy.animation import Animation
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.video import Image
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.clock import Clock
Change the position to a percentage (strictly, a proportion)
           of duration.

        :Parameters:
            `percent`: float or int
                Position to seek as a proportion of total duration, must
                be between 0-1.
            `precise`: bool, defaults to True
                Precise seeking is slower, but seeks to exact requested
                percent.

        .. warning::
            Calling seek() before the video is loaded has no effect.

        .. versionadded:: 1.2.0

        .. versionchanged:: 1.10.1
            The `precise` keyword argument has been added.
        