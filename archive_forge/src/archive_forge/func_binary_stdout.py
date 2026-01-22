import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def binary_stdout():
    """
    A sys.stdout that accepts bytes.
    """
    stdout = sys.stdout.buffer
    if sys.platform == 'win32':
        import msvcrt
        import os
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    return stdout