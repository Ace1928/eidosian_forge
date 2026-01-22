from fcntl import ioctl
import re
import six
import struct
import sys
from termios import TIOCGWINSZ, TCSADRAIN, tcsetattr, tcgetattr
import textwrap
import tty
from .prefs import Prefs
def epy_write(self, text):
    """
        Renders and print and epytext-formatted text on the console.
        """
    text = self.dedent(text)
    clean_text = ''
    for index in range(1, len(text)):
        if text[-index] == '\n':
            clean_text = text[:-index]
            if index != 1:
                clean_text += text[-index + 1:]
            break
    else:
        clean_text = text
    self.raw_write(clean_text, output=self._stdout)