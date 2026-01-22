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
def test_unregistering(self):
    err_transformer = ErrorTransformer()
    ip.ast_transformers.append(err_transformer)
    with self.assertWarnsRegex(UserWarning, 'It will be unregistered'):
        ip.run_cell('1 + 2')
    self.assertNotIn(err_transformer, ip.ast_transformers)