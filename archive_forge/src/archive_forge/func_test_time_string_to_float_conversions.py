import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
def test_time_string_to_float_conversions(self):
    payload_cls = receipt_formatters.ReceiptPayload
    original_time_str = utils.isotime(subsecond=True)
    time_obj = timeutils.parse_isotime(original_time_str)
    expected_time_float = (timeutils.normalize_time(time_obj) - datetime.datetime.utcfromtimestamp(0)).total_seconds()
    self.assertIsInstance(expected_time_float, float)
    actual_time_float = payload_cls._convert_time_string_to_float(original_time_str)
    self.assertIsInstance(actual_time_float, float)
    self.assertEqual(expected_time_float, actual_time_float)
    time_object = datetime.datetime.utcfromtimestamp(actual_time_float)
    expected_time_str = utils.isotime(time_object, subsecond=True)
    actual_time_str = payload_cls._convert_float_to_time_string(actual_time_float)
    self.assertEqual(expected_time_str, actual_time_str)