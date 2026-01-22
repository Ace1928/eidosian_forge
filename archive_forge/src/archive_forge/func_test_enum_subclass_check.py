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
def test_enum_subclass_check(self):

    def _test():

        class BrokenEnumField(fields.BaseEnumField):
            AUTO_TYPE = int
        BrokenEnumField()
    self.assertRaises(exception.EnumFieldInvalid, _test)