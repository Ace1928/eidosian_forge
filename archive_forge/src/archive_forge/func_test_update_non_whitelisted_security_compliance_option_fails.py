import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_non_whitelisted_security_compliance_option_fails(self):
    """Updating security compliance options through the API is not allowed.

        Requests to update anything in the security compliance group through
        the API should be Forbidden. This ensures that we are covering cases
        where the option being updated isn't in the white list.
        """
    group = 'security_compliance'
    option = 'lockout_failure_attempts'
    url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
    new_config = {group: {option: 1}}
    self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.patch(url, body={'config': new_config}, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())