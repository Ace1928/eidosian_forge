import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_bcrypt_sha256_not_truncate_password(self):
    self.config_fixture.config(strict_password_check=True)
    self.config_fixture.config(group='identity', password_hash_algorithm='bcrypt_sha256')
    password = '0' * 128
    password_verified = common_utils.verify_length_and_trunc_password(password)
    hashed = common_utils.hash_password(password)
    self.assertTrue(common_utils.check_password(password, hashed))
    self.assertEqual(password.encode('utf-8'), password_verified)