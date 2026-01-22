import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_security_compliance_password_regex_fails(self):
    """Make sure any updates to security compliance options fail."""
    group = 'security_compliance'
    option = 'password_regex'
    url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
    new_config = {group: {option: uuid.uuid4().hex}}
    self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())