import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_not_closed_quotes(self):
    self.assertRaises(ValueError, strutils.split_by_commas, '"ab","b""')