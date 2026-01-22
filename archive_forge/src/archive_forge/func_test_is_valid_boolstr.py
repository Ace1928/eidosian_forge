import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_is_valid_boolstr(self):
    self.assertTrue(strutils.is_valid_boolstr('true'))
    self.assertTrue(strutils.is_valid_boolstr('false'))
    self.assertTrue(strutils.is_valid_boolstr('yes'))
    self.assertTrue(strutils.is_valid_boolstr('no'))
    self.assertTrue(strutils.is_valid_boolstr('y'))
    self.assertTrue(strutils.is_valid_boolstr('n'))
    self.assertTrue(strutils.is_valid_boolstr('1'))
    self.assertTrue(strutils.is_valid_boolstr('0'))
    self.assertTrue(strutils.is_valid_boolstr(1))
    self.assertTrue(strutils.is_valid_boolstr(0))
    self.assertFalse(strutils.is_valid_boolstr('maybe'))
    self.assertFalse(strutils.is_valid_boolstr('only on tuesdays'))