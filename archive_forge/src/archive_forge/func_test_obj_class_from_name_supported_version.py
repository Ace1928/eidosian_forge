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
def test_obj_class_from_name_supported_version(self):
    self.assertRaises(exception.IncompatibleObjectVersion, base.VersionedObject.obj_class_from_name, 'MyObj', '1.25')
    try:
        base.VersionedObject.obj_class_from_name('MyObj', '1.25')
    except exception.IncompatibleObjectVersion as error:
        self.assertEqual('1.6', error.kwargs['supported'])