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
def test_In_variable(self):
    """Verify that In variable grows with user input (GH-284)"""
    oldlen = len(ip.user_ns['In'])
    ip.run_cell('1;', store_history=True)
    newlen = len(ip.user_ns['In'])
    self.assertEqual(oldlen + 1, newlen)
    self.assertEqual(ip.user_ns['In'][-1], '1;')