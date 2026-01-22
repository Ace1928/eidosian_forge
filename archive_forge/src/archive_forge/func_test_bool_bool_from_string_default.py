import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_bool_bool_from_string_default(self):
    self.assertTrue(strutils.bool_from_string('', default=True))
    self.assertFalse(strutils.bool_from_string('wibble', default=False))