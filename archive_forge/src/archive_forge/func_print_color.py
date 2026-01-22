from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
def print_color(s, color=None, bold=False, file=sys.stdout):
    """Print a colorized version of string."""
    if not term_supports_colors():
        print(s, file=file)
    elif POSIX:
        print(hilite(s, color, bold), file=file)
    else:
        import ctypes
        DEFAULT_COLOR = 7
        GetStdHandle = ctypes.windll.Kernel32.GetStdHandle
        SetConsoleTextAttribute = ctypes.windll.Kernel32.SetConsoleTextAttribute
        colors = dict(green=2, red=4, brown=6, yellow=6)
        colors[None] = DEFAULT_COLOR
        try:
            color = colors[color]
        except KeyError:
            raise ValueError('invalid color %r; choose between %r' % (color, list(colors.keys())))
        if bold and color <= 7:
            color += 8
        handle_id = -12 if file is sys.stderr else -11
        GetStdHandle.restype = ctypes.c_ulong
        handle = GetStdHandle(handle_id)
        SetConsoleTextAttribute(handle, color)
        try:
            print(s, file=file)
        finally:
            SetConsoleTextAttribute(handle, DEFAULT_COLOR)