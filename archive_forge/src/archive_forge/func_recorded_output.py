import contextlib
import fcntl
import itertools
import multiprocessing
import os
import pty
import re
import signal
import struct
import sys
import tempfile
import termios
import time
import traceback
import types
from typing import Optional, Generator, Tuple
import typing
def recorded_output(self) -> typing.Tuple[typing.List[str], typing.List[str]]:
    return (self._raw_lines[:], self._lines[:])