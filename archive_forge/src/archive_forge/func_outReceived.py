import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
def outReceived(self, data):
    self.data += data.decode(encoding)
    if self.delayed_call is not None:
        self.delayed_call.cancel()
    self.delayed_call = reactor.callLater(0.5, self.next)