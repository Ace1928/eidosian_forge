from fcntl import ioctl
import re
import six
import struct
import sys
from termios import TIOCGWINSZ, TCSADRAIN, tcsetattr, tcgetattr
import textwrap
import tty
from .prefs import Prefs
def get_cursor_xy(self):
    """
        Get the current text cursor x, y coordinates.
        """
    coords = [int(coord) for coord in self.escape('6n', 'R')]
    coords.reverse()
    return coords