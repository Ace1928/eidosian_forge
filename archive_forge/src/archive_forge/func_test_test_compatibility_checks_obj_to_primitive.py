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
def test_test_compatibility_checks_obj_to_primitive(self):
    fake = mock.MagicMock()
    fake.VERSION = '1.3'
    checker = fixture.ObjectVersionChecker()
    checker._test_object_compatibility(fake)
    fake().obj_to_primitive.assert_has_calls([mock.call(target_version='1.0'), mock.call(target_version='1.1'), mock.call(target_version='1.2'), mock.call(target_version='1.3')])