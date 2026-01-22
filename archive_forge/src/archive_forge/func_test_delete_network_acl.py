from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_delete_network_acl(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.delete_network_acl_entry('acl-2cb85d45', 100, egress=False)
    self.assert_request_parameters({'Action': 'DeleteNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 100, 'Egress': 'false'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response, True)