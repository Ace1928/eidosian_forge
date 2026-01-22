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
def test_get_exception_only(self):
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        msg = ip.get_exception_only()
    self.assertEqual(msg, 'KeyboardInterrupt\n')
    try:
        raise DerivedInterrupt('foo')
    except KeyboardInterrupt:
        msg = ip.get_exception_only()
    self.assertEqual(msg, 'IPython.core.tests.test_interactiveshell.DerivedInterrupt: foo\n')