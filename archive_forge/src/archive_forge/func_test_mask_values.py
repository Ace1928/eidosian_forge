import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_mask_values(self):
    payload = {'somekey': 'test = cmd --password myé\x80\x80pass'}
    expected = {'somekey': 'test = cmd --password ***'}
    self.assertEqual(expected, strutils.mask_dict_password(payload))