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
def test_verify_length_and_trunc_password_throws_validation_error(self):

    class SpecialObject(object):
        pass
    special_object = SpecialObject()
    invalid_passwords = [True, special_object, 4.3, 5]
    for invalid_password in invalid_passwords:
        self.assertRaises(exception.ValidationError, common_utils.verify_length_and_trunc_password, invalid_password)