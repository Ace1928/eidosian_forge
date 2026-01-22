import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_obj_make_compatible_on_list_base(self):

    @base.VersionedObjectRegistry.register_if(False)
    class MyList(base.ObjectListBase, base.VersionedObject):
        VERSION = '1.1'
        fields = {'objects': fields.ListOfObjectsField('MyObj')}
    childobj = MyObj(foo=1)
    listobj = MyList(objects=[childobj])
    compat_func = 'obj_make_compatible_from_manifest'
    with mock.patch.object(childobj, compat_func) as mock_compat:
        listobj.obj_to_primitive(target_version='1.0')
        mock_compat.assert_called_once_with({'foo': 1}, '1.0', version_manifest=None)