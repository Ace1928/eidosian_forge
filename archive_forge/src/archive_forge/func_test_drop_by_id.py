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
def test_drop_by_id(self):
    myvars = {'a': object(), 'b': object(), 'c': object()}
    ip.push(myvars, interactive=False)
    for name in myvars:
        assert name in ip.user_ns, name
        assert name in ip.user_ns_hidden, name
    ip.user_ns['b'] = 12
    ip.drop_by_id(myvars)
    for name in ['a', 'c']:
        assert name not in ip.user_ns, name
        assert name not in ip.user_ns_hidden, name
    assert ip.user_ns['b'] == 12
    ip.reset()