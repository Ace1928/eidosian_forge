import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
def test_bad_custom_tb(self):
    """Check that InteractiveShell is protected from bad custom exception handlers"""
    ip.set_custom_exc((IOError,), lambda etype, value, tb: 1 / 0)
    self.assertEqual(ip.custom_exceptions, (IOError,))
    with tt.AssertPrints('Custom TB Handler failed', channel='stderr'):
        ip.run_cell(u'raise IOError("foo")')
    self.assertEqual(ip.custom_exceptions, ())