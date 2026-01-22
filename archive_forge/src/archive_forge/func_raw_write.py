from fcntl import ioctl
import re
import six
import struct
import sys
from termios import TIOCGWINSZ, TCSADRAIN, tcsetattr, tcgetattr
import textwrap
import tty
from .prefs import Prefs
def raw_write(self, text, output=sys.stdout):
    """
        Raw console printing function.
        @param text: The text to print.
        @type text: str
        """
    output.write(text)
    output.flush()