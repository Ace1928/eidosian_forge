from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
@contextmanager
def hidden_cursor(self):
    """Return a context manager that hides the cursor while inside it and
        makes it visible on leaving."""
    self.stream.write(self.hide_cursor)
    try:
        yield
    finally:
        self.stream.write(self.normal_cursor)