import os
import sys
from os.path import pardir, realpath
def joinuser(*args):
    return os.path.expanduser(os.path.join(*args))