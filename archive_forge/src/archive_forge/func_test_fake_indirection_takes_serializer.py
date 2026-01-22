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
def test_fake_indirection_takes_serializer(self):
    ser = mock.MagicMock()
    iapi = fixture.FakeIndirectionAPI(ser)
    ser.serialize_entity.return_value = mock.sentinel.serial
    iapi.object_action(mock.sentinel.context, mock.sentinel.objinst, mock.sentinel.objmethod, (), {})
    ser.serialize_entity.assert_called_once_with(mock.sentinel.context, mock.sentinel.objinst)
    ser.deserialize_entity.assert_called_once_with(mock.sentinel.context, mock.sentinel.serial)