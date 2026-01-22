import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def setpassword(self, pwd):
    """Set default password for encrypted files."""
    if pwd and (not isinstance(pwd, bytes)):
        raise TypeError('pwd: expected bytes, got %s' % type(pwd).__name__)
    if pwd:
        self.pwd = pwd
    else:
        self.pwd = None