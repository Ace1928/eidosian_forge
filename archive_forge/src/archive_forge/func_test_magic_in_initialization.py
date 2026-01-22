from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_in_initialization(self):
    m = MagicMock(**{'__str__.return_value': '12'})
    self.assertEqual(str(m), '12')