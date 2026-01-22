import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_non_whitelisted_security_compliance_opt_fails(self):
    """We only support exposing a subset of security compliance options.

        Given that security compliance information is sensitive in nature, we
        should make sure that only the options we want to expose are readable
        via the API.
        """
    self.config_fixture.config(group='security_compliance', lockout_failure_attempts=1)
    url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance', 'option': 'lockout_failure_attempts'}
    self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())
    self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())