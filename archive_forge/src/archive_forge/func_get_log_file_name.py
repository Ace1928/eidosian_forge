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
def get_log_file_name(level=INFO):
    """Returns the name of the log file.

  For Python logging, only one file is used and level is ignored. And it returns
  empty string if it logs to stderr/stdout or the log stream has no `name`
  attribute.

  Args:
    level: int, the absl.logging level.

  Raises:
    ValueError: Raised when `level` has an invalid value.
  """
    if level not in converter.ABSL_LEVELS:
        raise ValueError('Invalid absl.logging level {}'.format(level))
    stream = get_absl_handler().python_handler.stream
    if stream == sys.stderr or stream == sys.stdout or (not hasattr(stream, 'name')):
        return ''
    else:
        return stream.name