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
def test_ofind_slotted_attributes(self):

    class A(object):
        __slots__ = ['foo']

        def __init__(self):
            self.foo = 'bar'
    a = A()
    found = ip._ofind('a.foo', [('locals', locals())])
    info = OInfo(found=True, isalias=False, ismagic=False, namespace='locals', obj=a.foo, parent=a)
    self.assertEqual(found, info)
    found = ip._ofind('a.bar', [('locals', locals())])
    expected = OInfo(found=False, isalias=False, ismagic=False, namespace=None, obj=None, parent=a)
    assert found == expected