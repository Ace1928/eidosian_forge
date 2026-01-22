from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_changing_magic_set_in_initialization(self):
    m = MagicMock(**{'__str__.return_value': '12'})
    m.__str__.return_value = '13'
    self.assertEqual(str(m), '13')
    m = MagicMock(**{'__str__.return_value': '12'})
    m.configure_mock(**{'__str__.return_value': '14'})
    self.assertEqual(str(m), '14')