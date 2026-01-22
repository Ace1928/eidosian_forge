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
def test_custom_syntaxerror_exception(self):
    called = []

    def my_handler(shell, etype, value, tb, tb_offset=None):
        called.append(etype)
        shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
    ip.set_custom_exc((SyntaxError,), my_handler)
    try:
        ip.run_cell('1f')
        self.assertEqual(called, [SyntaxError])
    finally:
        ip.set_custom_exc((), None)