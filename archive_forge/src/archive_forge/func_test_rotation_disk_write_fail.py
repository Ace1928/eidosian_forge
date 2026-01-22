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
def test_rotation_disk_write_fail(self):
    self.assertRepositoryState(expected_size=2)
    key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
    mock_open = mock.mock_open()
    file_handle = mock_open()
    file_handle.flush.side_effect = IOError('disk full')
    with mock.patch('keystone.common.fernet_utils.open', mock_open):
        self.assertRaises(IOError, key_utils.rotate_keys)
    self.assertEqual(self.key_repository_size, 2)
    with mock.patch('keystone.common.fernet_utils.open', mock_open):
        self.assertRaises(IOError, key_utils.rotate_keys)
    self.assertEqual(self.key_repository_size, 2)
    key_utils.rotate_keys()
    self.assertEqual(self.key_repository_size, 3)