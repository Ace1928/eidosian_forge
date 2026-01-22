import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_domain_config(self):
    config_ref = self._new_ref()
    config = self.client.domain_configs.create(self.test_domain.id, config_ref)
    self.addCleanup(self.client.domain_configs.delete, self.test_domain.id)
    self.check_domain_config(config, config_ref)