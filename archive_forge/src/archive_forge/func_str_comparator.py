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
def str_comparator(self, expected, obj_val):
    """Compare a field to a string value

        Compare an object field to a string in the db by performing
        a simple coercion on the object field value.
        """
    self.assertEqual(expected, str(obj_val))