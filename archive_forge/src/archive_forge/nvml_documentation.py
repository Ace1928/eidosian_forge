from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string

        Maps value to a proper subclass of NVMLError.
        See _extractNVMLErrorsAsClasses function for more details
        