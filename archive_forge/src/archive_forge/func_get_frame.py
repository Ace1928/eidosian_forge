import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def get_frame(level=0):
    """Return frame object from call stack with given level.
    """
    try:
        return sys._getframe(level + 1)
    except AttributeError:
        frame = sys.exc_info()[2].tb_frame
        for _ in range(level + 1):
            frame = frame.f_back
        return frame