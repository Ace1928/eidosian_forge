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
@onlyif_cmds_exist('csh')
def test_exit_code_signal_csh(self):
    SHELL = os.environ.get('SHELL', None)
    os.environ['SHELL'] = find_cmd('csh')
    try:
        self.test_exit_code_signal()
    finally:
        if SHELL is not None:
            os.environ['SHELL'] = SHELL
        else:
            del os.environ['SHELL']