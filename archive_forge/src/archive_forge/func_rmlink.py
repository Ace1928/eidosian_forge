import errno
import os
from time import time as _uniquefloat
from twisted.python.runtime import platform
from os import rename
def rmlink(filename):
    os.remove(os.path.join(filename, 'symlink'))
    os.rmdir(filename)