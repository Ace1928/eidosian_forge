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

        Run bpython (with `backend` as backend) in a subprocess and
        enter the given input. Uses a test config that disables the
        paste detection.

        Returns bpython's output.
        