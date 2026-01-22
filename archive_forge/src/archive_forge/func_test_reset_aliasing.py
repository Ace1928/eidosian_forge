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
def test_reset_aliasing(self):
    """ Check that standard posix aliases work after %reset. """
    if os.name != 'posix':
        return
    ip.reset()
    for cmd in ('clear', 'more', 'less', 'man'):
        res = ip.run_cell('%' + cmd)
        self.assertEqual(res.success, True)