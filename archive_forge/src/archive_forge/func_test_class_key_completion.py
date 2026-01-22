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
def test_class_key_completion(self):
    ip = get_ipython()
    NamedInstanceClass('qwerty')
    NamedInstanceClass('qwick')
    ip.user_ns['named_instance_class'] = NamedInstanceClass
    _, matches = ip.Completer.complete(line_buffer="named_instance_class['qw")
    self.assertIn('qwerty', matches)
    self.assertIn('qwick', matches)