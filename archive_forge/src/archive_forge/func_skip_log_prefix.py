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
def skip_log_prefix(func):
    """Skips reporting the prefix of a given function or name by ABSLLogger.

  This is a convenience wrapper function / decorator for
  `ABSLLogger.register_frame_to_skip`.

  If a callable function is provided, only that function will be skipped.
  If a function name is provided, all functions with the same name in the
  file that this is called in will be skipped.

  This can be used as a decorator of the intended function to be skipped.

  Args:
    func: Callable function or its name as a string.

  Returns:
    func (the input, unchanged).

  Raises:
    ValueError: The input is callable but does not have a function code object.
    TypeError: The input is neither callable nor a string.
  """
    if callable(func):
        func_code = getattr(func, '__code__', None)
        if func_code is None:
            raise ValueError('Input callable does not have a function code object.')
        file_name = func_code.co_filename
        func_name = func_code.co_name
        func_lineno = func_code.co_firstlineno
    elif isinstance(func, six.string_types):
        file_name = get_absl_logger().findCaller()[0]
        func_name = func
        func_lineno = None
    else:
        raise TypeError('Input is neither callable nor a string.')
    ABSLLogger.register_frame_to_skip(file_name, func_name, func_lineno)
    return func