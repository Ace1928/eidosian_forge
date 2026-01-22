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
def test_var_expand(self):
    ip.user_ns['f'] = u'Caño'
    self.assertEqual(ip.var_expand(u'echo $f'), u'echo Caño')
    self.assertEqual(ip.var_expand(u'echo {f}'), u'echo Caño')
    self.assertEqual(ip.var_expand(u'echo {f[:-1]}'), u'echo Cañ')
    self.assertEqual(ip.var_expand(u'echo {1*2}'), u'echo 2')
    self.assertEqual(ip.var_expand(u"grep x | awk '{print $1}'"), u"grep x | awk '{print $1}'")
    ip.user_ns['f'] = b'Ca\xc3\xb1o'
    ip.var_expand(u'echo $f')