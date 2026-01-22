import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_security_compliance_config_for_default_domain(self):
    """Ask for all security compliance configuration options.

        Support for enforcing security compliance per domain currently doesn't
        exist. Make sure when we ask for security compliance information, it's
        only for the default domain and that it only returns whitelisted
        options.
        """
    password_regex = uuid.uuid4().hex
    password_regex_description = uuid.uuid4().hex
    self.config_fixture.config(group='security_compliance', password_regex=password_regex)
    self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
    expected_response = {'security_compliance': {'password_regex': password_regex, 'password_regex_description': password_regex_description}}
    url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance'}
    regular_response = self.get(url, token=self._get_non_admin_token())
    self.assertEqual(regular_response.result['config'], expected_response)
    admin_response = self.get(url, token=self._get_admin_token())
    self.assertEqual(admin_response.result['config'], expected_response)
    self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
    self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)