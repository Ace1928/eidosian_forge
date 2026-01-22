from unittest import mock
import ddt
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import security_services
def test_create_all_fields(self):
    values = {'type': 'ldap', 'dns_ip': 'fake dns ip', 'ou': 'fake ou', 'server': 'fake.ldap.server', 'domain': 'fake.ldap.domain', 'user': 'fake user', 'password': 'fake password', 'name': 'fake name', 'description': 'fake description'}
    with mock.patch.object(self.manager, '_create', fakes.fake_create):
        result = self.manager.create(**values)
        self.assertEqual(result['url'], security_services.RESOURCES_PATH)
        self.assertEqual(result['resp_key'], security_services.RESOURCE_NAME)
        self.assertIn(security_services.RESOURCE_NAME, result['body'])
        self.assertEqual(result['body'][security_services.RESOURCE_NAME], values)