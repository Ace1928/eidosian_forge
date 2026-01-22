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
def use_absl_handler():
    """Uses the ABSL logging handler for logging.

  This method is called in app.run() so the absl handler is used in absl apps.
  """
    global _attempted_to_remove_stderr_stream_handlers
    if not _attempted_to_remove_stderr_stream_handlers:
        handlers = [h for h in logging.root.handlers if isinstance(h, logging.StreamHandler) and h.stream == sys.stderr]
        for h in handlers:
            logging.root.removeHandler(h)
        _attempted_to_remove_stderr_stream_handlers = True
    absl_handler = get_absl_handler()
    if absl_handler not in logging.root.handlers:
        logging.root.addHandler(absl_handler)
        FLAGS['verbosity']._update_logging_levels()
        FLAGS['logger_levels']._update_logger_levels()