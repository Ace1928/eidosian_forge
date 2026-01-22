import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_default_by_group(self):
    """Call ``GET & HEAD /domains/config/{group}/default``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    url = '/domains/config/ldap/default'
    r = self.get(url)
    default_config = r.result['config']
    for option in default_config['ldap']:
        self.assertEqual(getattr(CONF.ldap, option), default_config['ldap'][option])
    self.head(url, expected_status=http.client.OK)