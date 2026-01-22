import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_config(self):
    """Call ``DELETE /domains{domain_id}/config``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    self.delete('/domains/%(domain_id)s/config' % {'domain_id': self.domain['id']})
    self.get('/domains/%(domain_id)s/config' % {'domain_id': self.domain['id']}, expected_status=exception.DomainConfigNotFound.code)