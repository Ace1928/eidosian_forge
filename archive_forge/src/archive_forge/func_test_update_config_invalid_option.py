import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_config_invalid_option(self):
    """Call ``PATCH /domains/{domain_id}/config/{group}/{invalid}``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    invalid_option = uuid.uuid4().hex
    new_config = {'ldap': {invalid_option: uuid.uuid4().hex}}
    self.patch('/domains/%(domain_id)s/config/ldap/%(invalid_option)s' % {'domain_id': self.domain['id'], 'invalid_option': invalid_option}, body={'config': new_config}, expected_status=http.client.FORBIDDEN)
    new_config = {'suffix': uuid.uuid4().hex}
    self.patch('/domains/%(domain_id)s/config/ldap/suffix' % {'domain_id': self.domain['id']}, body={'config': new_config}, expected_status=http.client.NOT_FOUND)