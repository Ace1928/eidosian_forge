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
def test_run_empty_cell(self):
    """Just make sure we don't get a horrible error with a blank
        cell of input. Yes, I did overlook that."""
    old_xc = ip.execution_count
    res = ip.run_cell('')
    self.assertEqual(ip.execution_count, old_xc)
    self.assertEqual(res.execution_count, None)