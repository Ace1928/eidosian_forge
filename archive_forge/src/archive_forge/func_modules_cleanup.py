from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def modules_cleanup(oldmodules):
    encodings = [(k, v) for k, v in sys.modules.items() if k.startswith('encodings.')]
    for i in range(len(sys.modules)):
        sys.modules.pop()
    sys.modules.update(encodings)
    sys.modules.update(oldmodules)