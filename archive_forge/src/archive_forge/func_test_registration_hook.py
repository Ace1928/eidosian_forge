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
def test_registration_hook(self):

    class TestObject(base.VersionedObject):
        VERSION = '1.0'

    class TestObjectNewer(base.VersionedObject):
        VERSION = '1.1'

        @classmethod
        def obj_name(cls):
            return 'TestObject'
    registry = base.VersionedObjectRegistry()
    with mock.patch.object(registry, 'registration_hook') as mock_hook:
        registry._register_class(TestObject)
        mock_hook.assert_called_once_with(TestObject, 0)
    with mock.patch.object(registry, 'registration_hook') as mock_hook:
        registry._register_class(TestObjectNewer)
        mock_hook.assert_called_once_with(TestObjectNewer, 0)