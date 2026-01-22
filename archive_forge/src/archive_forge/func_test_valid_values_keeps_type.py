import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
def test_valid_values_keeps_type(self):
    self.assertIsInstance(self.field.valid_values, tuple)
    self.assertIsInstance(FakeEnumAltField().valid_values, set)