import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_config_twice(self):
    """Check multiple creates don't throw error."""
    self.put('/domains/%(domain_id)s/config' % {'domain_id': self.domain['id']}, body={'config': self.config}, expected_status=http.client.CREATED)
    self.put('/domains/%(domain_id)s/config' % {'domain_id': self.domain['id']}, body={'config': self.config}, expected_status=http.client.OK)