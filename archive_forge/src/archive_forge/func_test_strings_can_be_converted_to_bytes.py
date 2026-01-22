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
def test_strings_can_be_converted_to_bytes(self):
    s = token_provider.random_urlsafe_str()
    self.assertIsInstance(s, str)
    b = receipt_formatters.ReceiptPayload.random_urlsafe_str_to_bytes(s)
    self.assertIsInstance(b, bytes)