from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_create_network_acl(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.create_network_acl_entry('acl-2cb85d45', 110, 'udp', 'allow', '0.0.0.0/0', egress=False, port_range_from=53, port_range_to=53)
    self.assert_request_parameters({'Action': 'CreateNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 110, 'Protocol': 'udp', 'RuleAction': 'allow', 'Egress': 'false', 'CidrBlock': '0.0.0.0/0', 'PortRange.From': 53, 'PortRange.To': 53}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response, True)