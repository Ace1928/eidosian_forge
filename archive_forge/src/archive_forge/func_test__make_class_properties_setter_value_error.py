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
@mock.patch.object(obj_base.LOG, 'exception')
def test__make_class_properties_setter_value_error(self, mock_log):

    @obj_base.VersionedObjectRegistry.register
    class AnObject(obj_base.VersionedObject):
        fields = {'intfield': fields.IntegerField()}
    self.assertRaises(ValueError, AnObject, intfield='badvalue')
    self.assertFalse(mock_log.called)