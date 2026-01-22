import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_config_invalid_group_invalid_domain(self):
    """Call ``PATCH /domains/{domain_id}/config/{invalid_group}``.

        While updating Identity API-based domain config with an invalid group
        and an invalid domain id provided, the request shall be rejected
        with a response, 404 domain not found.
        """
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    invalid_group = uuid.uuid4().hex
    new_config = {invalid_group: {'url': uuid.uuid4().hex, 'user_filter': uuid.uuid4().hex}}
    invalid_domain_id = uuid.uuid4().hex
    self.patch('/domains/%(domain_id)s/config/%(invalid_group)s' % {'domain_id': invalid_domain_id, 'invalid_group': invalid_group}, body={'config': new_config}, expected_status=exception.DomainNotFound.code)