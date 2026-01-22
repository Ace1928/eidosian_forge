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
def find_helper_script(filedir, script_name):
    target_filename = os.path.join(filedir, 'linux_and_mac', script_name)
    target_filename = os.path.normpath(target_filename)
    if not os.path.exists(target_filename):
        raise RuntimeError('Could not find helper script: %s' % target_filename)
    return target_filename