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
def test_var_expand_local(self):
    """Test local variable expansion in !system and %magic calls"""
    ip.run_cell('def test():\n    lvar = "ttt"\n    ret = !echo {lvar}\n    return ret[0]\n')
    res = ip.user_ns['test']()
    self.assertIn('ttt', res)
    ip.run_cell('def makemacro():\n    macroname = "macro_var_expand_locals"\n    %macro {macroname} codestr\n')
    ip.user_ns['codestr'] = 'str(12)'
    ip.run_cell('makemacro()')
    self.assertIn('macro_var_expand_locals', ip.user_ns)