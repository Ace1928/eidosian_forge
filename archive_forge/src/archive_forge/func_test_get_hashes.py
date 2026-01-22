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
def test_get_hashes(self):
    checker = fixture.ObjectVersionChecker()
    hashes = checker.get_hashes()
    self.assertEqual('1.6-fb5f5379168bf08f7f2ce0a745e91027', hashes['TestSubclassedObject'])