import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_dict_key_completion_invalids(self):
    """Smoke test cases dict key completion can't handle"""
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['no_getitem'] = None
    ip.user_ns['no_keys'] = []
    ip.user_ns['cant_call_keys'] = dict
    ip.user_ns['empty'] = {}
    ip.user_ns['d'] = {'abc': 5}
    _, matches = complete(line_buffer="no_getitem['")
    _, matches = complete(line_buffer="no_keys['")
    _, matches = complete(line_buffer="cant_call_keys['")
    _, matches = complete(line_buffer="empty['")
    _, matches = complete(line_buffer="name_error['")
    _, matches = complete(line_buffer="d['\\")