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
def test_run_cell(self):
    ip.run_cell('n = 2')
    self.assertEqual(self.calls, [(2,)])
    ip.run_cell('o = 2.0')
    self.assertEqual(ip.user_ns['o'], 2.0)