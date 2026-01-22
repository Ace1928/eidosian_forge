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
def test_indirection_action(self):
    self.useFixture(fixture.IndirectionFixture())
    obj = MyObj(context=self.context)
    with mock.patch.object(base.VersionedObject.indirection_api, 'object_action') as mock_action:
        mock_action.return_value = ({}, 'foo')
        obj.marco()
        mock_action.assert_called_once_with(self.context, obj, 'marco', (), {})