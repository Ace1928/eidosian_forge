from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_replace_network_acl_icmp(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.replace_network_acl_entry('acl-2cb85d45', 110, 'tcp', 'deny', '0.0.0.0/0', icmp_code=-1, icmp_type=8)
    self.assert_request_parameters({'Action': 'ReplaceNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 110, 'Protocol': 'tcp', 'RuleAction': 'deny', 'CidrBlock': '0.0.0.0/0', 'Icmp.Code': -1, 'Icmp.Type': 8}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response, True)