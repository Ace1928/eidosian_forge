import copy
import functools
import getpass
import logging
import os
import time
import warnings
from cliff import columns as cliff_columns
from oslo_utils import importutils
from osc_lib import exceptions
from osc_lib.i18n import _
def get_effective_log_level():
    """Returns the lowest logging level considered by logging handlers

    Retrieve and return the smallest log level set among the root
    logger's handlers (in case of multiple handlers).
    """
    root_log = logging.getLogger()
    min_log_lvl = logging.CRITICAL
    for handler in root_log.handlers:
        min_log_lvl = min(min_log_lvl, handler.level)
    return min_log_lvl