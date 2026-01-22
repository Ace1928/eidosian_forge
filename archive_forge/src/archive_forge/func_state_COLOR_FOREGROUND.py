import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def state_COLOR_FOREGROUND(self, ch):
    """
        Handle the foreground color state.

        Foreground colors can consist of up to two digits and may optionally
        end in a I{,}. Any non-digit or non-comma characters are treated as
        invalid input and result in the state being reset to "text".

        @param ch: The character being processed.
        """
    if ch.isdigit() and len(self._buffer) < 2:
        self._buffer += ch
    else:
        if self._buffer:
            col = int(self._buffer) % len(_IRC_COLORS)
            self.foreground = getattr(attributes.fg, _IRC_COLOR_NAMES[col])
        else:
            self.foreground = self.background = None
        if ch == ',' and self._buffer:
            self._buffer = ''
            self.state = 'COLOR_BACKGROUND'
        else:
            self._buffer = ''
            self.state = 'TEXT'
            self.emit()
            self.process(ch)