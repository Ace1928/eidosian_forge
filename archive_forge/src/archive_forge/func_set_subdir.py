import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def set_subdir(self, subdir):
    """Set subdir to 'new' or 'cur'."""
    if subdir == 'new' or subdir == 'cur':
        self._subdir = subdir
    else:
        raise ValueError("subdir must be 'new' or 'cur': %s" % subdir)