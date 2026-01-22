import sys
from distutils.log import *  # noqa: F403
from distutils.log import Log as old_Log
from distutils.log import _global_log
from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
def get_threshold():
    return _global_log.threshold