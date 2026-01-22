from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def useImageNotFoundException(value=None):
    """
    When called with no arguments, PyAutoGUI will raise ImageNotFoundException when the PyScreeze locate*() functions
    can't find the image it was told to locate. The default behavior is to return None. Call this function with no
    arguments (or with True as the argument) to have exceptions raised, which is a better practice.

    You can also disable raising exceptions by passing False for the argument.
    """
    if value is None:
        value = True
    try:
        pyscreeze.USE_IMAGE_NOT_FOUND_EXCEPTION = value
    except NameError:
        raise PyAutoGUIException("useImageNotFoundException() ws called but pyscreeze isn't installed.")