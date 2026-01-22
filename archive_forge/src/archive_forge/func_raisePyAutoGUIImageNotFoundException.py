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
def raisePyAutoGUIImageNotFoundException(wrappedFunction):
    """
    A decorator that wraps PyScreeze locate*() functions so that the PyAutoGUI user sees them raise PyAutoGUI's
    ImageNotFoundException rather than PyScreeze's ImageNotFoundException. This is because PyScreeze should be
    invisible to PyAutoGUI users.
    """

    @functools.wraps(wrappedFunction)
    def wrapper(*args, **kwargs):
        try:
            return wrappedFunction(*args, **kwargs)
        except pyscreeze.ImageNotFoundException:
            raise ImageNotFoundException
    return wrapper