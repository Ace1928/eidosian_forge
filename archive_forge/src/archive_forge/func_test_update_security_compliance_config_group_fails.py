import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_security_compliance_config_group_fails(self):
    """Make sure that updates to the entire security group section fail.

        We should only allow the ability to modify a deployments security
        compliance rules through configuration. Especially since it's only
        enforced on the default domain.
        """
    new_config = {'security_compliance': {'password_regex': uuid.uuid4().hex, 'password_regex_description': uuid.uuid4().hex}}
    url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance'}
    self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())