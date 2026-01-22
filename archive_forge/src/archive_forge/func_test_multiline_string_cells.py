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
def test_multiline_string_cells(self):
    """Code sprinkled with multiline strings should execute (GH-306)"""
    ip.run_cell('tmp=0')
    self.assertEqual(ip.user_ns['tmp'], 0)
    res = ip.run_cell('tmp=1;"""a\nb"""\n')
    self.assertEqual(ip.user_ns['tmp'], 1)
    self.assertEqual(res.success, True)
    self.assertEqual(res.result, 'a\nb')