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
def test_url_safe_with_unicode_check(self):
    base_str = u'i am çafe'
    self.assertFalse(common_utils.is_not_url_safe(base_str))
    for i in common_utils.URL_RESERVED_CHARS:
        self.assertTrue(common_utils.is_not_url_safe(base_str + i))