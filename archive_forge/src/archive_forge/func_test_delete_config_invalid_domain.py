import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_config_invalid_domain(self):
    """Call ``DELETE /domains{domain_id}/config``.

        While deleting Identity API-based domain config with an invalid domain
        id provided, the request shall be rejected with a response, 404 domain
        not found.
        """
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    invalid_domain_id = uuid.uuid4().hex
    self.delete('/domains/%(domain_id)s/config' % {'domain_id': invalid_domain_id}, expected_status=exception.DomainNotFound.code)