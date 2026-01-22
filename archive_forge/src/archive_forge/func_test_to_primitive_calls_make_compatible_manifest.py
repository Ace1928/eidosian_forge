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
def test_to_primitive_calls_make_compatible_manifest(self):
    obj = self.ParentObj()
    with mock.patch.object(obj, 'obj_make_compatible_from_manifest') as m:
        obj.obj_to_primitive(target_version='1.0', version_manifest=mock.sentinel.manifest)
        m.assert_called_once_with(mock.ANY, '1.0', mock.sentinel.manifest)