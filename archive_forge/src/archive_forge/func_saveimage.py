from threading import RLock
from sympy.core.numbers import Integer
from sympy.external.gmpy import SYMPY_INTS
from sympy.geometry.entity import GeometryEntity
from sympy.plotting.pygletplot.plot_axes import PlotAxes
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.plot_window import PlotWindow
from sympy.plotting.pygletplot.util import parse_option_string
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import is_sequence
from time import sleep
from os import getcwd, listdir
import ctypes
def saveimage(self, outfile=None, format='', size=(600, 500)):
    """
        Saves a screen capture of the plot window to an
        image file.

        If outfile is given, it can either be a path
        or a file object. Otherwise a png image will
        be saved to the current working directory.
        If the format is omitted, it is determined from
        the filename extension.
        """
    self._screenshot.save(outfile, format, size)