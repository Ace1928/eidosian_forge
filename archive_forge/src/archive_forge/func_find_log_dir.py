from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
def find_log_dir(log_dir=None):
    """Returns the most suitable directory to put log files into.

  Args:
    log_dir: str|None, if specified, the logfile(s) will be created in that
        directory.  Otherwise if the --log_dir command-line flag is provided,
        the logfile will be created in that directory.  Otherwise the logfile
        will be created in a standard location.

  Raises:
    FileNotFoundError: raised in Python 3 when it cannot find a log directory.
    OSError: raised in Python 2 when it cannot find a log directory.
  """
    if log_dir:
        dirs = [log_dir]
    elif FLAGS['log_dir'].value:
        dirs = [FLAGS['log_dir'].value]
    else:
        dirs = ['/tmp/', './']
    for d in dirs:
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    exception_class = OSError if six.PY2 else FileNotFoundError
    raise exception_class("Can't find a writable directory for logs, tried %s" % dirs)