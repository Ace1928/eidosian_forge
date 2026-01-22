from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_security_group_create_port_errors(self):
    self.requests_mock.register_uri('POST', FAKE_URL + '/os-security-group-rules', json={'security_group_rule': self.FAKE_SECURITY_GROUP_RULE_RESP}, status_code=200)
    self.assertRaises(compute.InvalidValue, self.api.security_group_rule_create, security_group_id='1', ip_protocol='tcp', from_port='', to_port=22, remote_ip='1.2.3.4/24')
    self.assertRaises(compute.InvalidValue, self.api.security_group_rule_create, security_group_id='1', ip_protocol='tcp', from_port=0, to_port=[], remote_ip='1.2.3.4/24')