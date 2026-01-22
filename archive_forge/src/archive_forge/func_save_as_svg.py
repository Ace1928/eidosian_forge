from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def save_as_svg(self, file_name, colormode='color', width=None):
    """
        The colormode (currently ignored) must be 'color', 'gray', or 'mono'.
        The width option is ignored for svg images.
        """
    save_as_svg(self.canvas, file_name, colormode, width)