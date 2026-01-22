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
def test_var_expand_self(self):
    """Test variable expansion with the name 'self', which was failing.
        
        See https://github.com/ipython/ipython/issues/1878#issuecomment-7698218
        """
    ip.run_cell('class cTest:\n  classvar="see me"\n  def test(self):\n    res = !echo Variable: {self.classvar}\n    return res[0]\n')
    self.assertIn('see me', ip.user_ns['cTest']().test())