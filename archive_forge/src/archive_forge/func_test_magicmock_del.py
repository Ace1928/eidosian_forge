from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magicmock_del(self):
    mock = MagicMock()
    del mock.__getitem__
    self.assertRaises(TypeError, lambda: mock['foo'])
    mock = MagicMock()
    mock['foo']
    del mock.__getitem__
    self.assertRaises(TypeError, lambda: mock['foo'])