import ctypes
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
import platform
import traceback
import os, time, sys
def run_python_code(*args, **kwargs):
    print('Unable to attach to process in platform: %s', sys.platform)