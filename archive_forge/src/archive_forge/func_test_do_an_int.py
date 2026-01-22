import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_do_an_int(self):
    payload = {}
    payload[1] = 2
    expected = payload.copy()
    self.assertEqual(expected, strutils.mask_dict_password(payload))