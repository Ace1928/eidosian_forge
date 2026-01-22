import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_default_by_option(self):
    """Call ``GET & HEAD /domains/config/{group}/{option}/default``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    url = '/domains/config/ldap/url/default'
    r = self.get(url)
    default_config = r.result['config']
    self.assertEqual(CONF.ldap.url, default_config['url'])
    self.head(url, expected_status=http.client.OK)