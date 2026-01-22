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
def test_do_subobject_backport_list_object(self):
    child = self.ChildObj(foo=1)
    parent = self.ParentObjList(objects=[child])
    parent_primitive = parent.obj_to_primitive()['versioned_object.data']
    primitive = child.obj_to_primitive()['versioned_object.data']
    version = '1.0'
    compat_func = 'obj_make_compatible_from_manifest'
    with mock.patch.object(child, compat_func) as mock_compat:
        base._do_subobject_backport(version, parent, 'objects', parent_primitive)
        mock_compat.assert_called_once_with(primitive, version, version_manifest=None)