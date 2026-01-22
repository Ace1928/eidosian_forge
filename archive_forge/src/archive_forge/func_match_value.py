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
def match_value(self, k, dv, v):
    """
        Try to match a single stored value (dv) with a supplied value (v).
        """
    if type(v) != type(dv):
        result = False
    elif type(dv) is not str or k not in self._partial_matches:
        result = v == dv
    else:
        result = dv.find(v) >= 0
    return result