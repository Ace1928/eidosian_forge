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
def handle_terminate(signum: int, frame: Optional[types.FrameType]) -> None:
    if p.is_alive():
        p.terminate()