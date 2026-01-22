from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_divmod_and_rdivmod(self):
    m = MagicMock()
    self.assertIsInstance(divmod(5, m), MagicMock)
    m.__divmod__.return_value = (2, 1)
    self.assertEqual(divmod(m, 2), (2, 1))
    m = MagicMock()
    foo = divmod(2, m)
    self.assertIsInstance(foo, MagicMock)
    foo_direct = m.__divmod__(2)
    self.assertIsInstance(foo_direct, MagicMock)
    bar = divmod(m, 2)
    self.assertIsInstance(bar, MagicMock)
    bar_direct = m.__rdivmod__(2)
    self.assertIsInstance(bar_direct, MagicMock)