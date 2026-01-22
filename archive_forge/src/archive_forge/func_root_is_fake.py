import sys
import doctest
import re
import types
from .numeric_output_checker import NumericOutputChecker
def root_is_fake():
    return _gui_status['fake_root']