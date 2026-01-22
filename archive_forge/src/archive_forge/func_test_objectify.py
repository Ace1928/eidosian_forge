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
@mock.patch.object(base.VersionedObjectRegistry, 'register_if')
def test_objectify(self, mock_register_if):
    mock_reg_callable = mock.Mock()
    mock_register_if.return_value = mock_reg_callable

    class my_class(object):
        pass
    base.VersionedObjectRegistry.objectify(my_class)
    mock_register_if.assert_called_once_with(False)
    mock_reg_callable.assert_called_once_with(my_class)