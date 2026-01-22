import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def handle_window_size_change(self):
    try:
        win_size = fcntl.ioctl(0, termios.TIOCGWINSZ, '\x00' * 8)
        rows, cols, xpix, ypix = struct.unpack('HHHH', win_size)
    except OSError:
        return
    if cols == 0:
        return
    win_size = struct.pack('HHHH', rows, cols, xpix, ypix)
    for fd in self._fds:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, win_size)