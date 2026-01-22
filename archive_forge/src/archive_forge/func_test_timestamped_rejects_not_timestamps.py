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
def test_timestamped_rejects_not_timestamps(self):
    with testtools.ExpectedException(ValueError, '.*parse date.*'):
        self.my_object.updated_at = 'a string'
    with testtools.ExpectedException(ValueError, '.*parse date.*'):
        self.my_object.created_at = 'a string'