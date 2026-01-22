import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_default(self):
    """Call ``GET & HEAD /domains/config/default``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    url = '/domains/config/default'
    r = self.get(url)
    default_config = r.result['config']
    for group in default_config:
        for option in default_config[group]:
            self.assertEqual(getattr(getattr(CONF, group), option), default_config[group][option])
    self.head(url, expected_status=http.client.OK)