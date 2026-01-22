import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_default_for_invalid_option(self):
    """Returning invalid configuration options is invalid."""
    url = '/domains/config/ldap/%s/default' % uuid.uuid4().hex
    self.get(url, expected_status=http.client.FORBIDDEN)
    self.head(url, expected_status=http.client.FORBIDDEN)