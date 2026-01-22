from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_disassociate_network_acl(self):
    self.https_connection.getresponse.side_effect = [self.create_response(status_code=200, body=self.get_all_network_acls_vpc_body), self.create_response(status_code=200, body=self.get_all_network_acls_subnet_body), self.create_response(status_code=200)]
    response = self.service_connection.disassociate_network_acl('subnet-ff669596', 'vpc-5266953b')
    self.assert_request_parameters({'Action': 'ReplaceNetworkAclAssociation', 'NetworkAclId': 'acl-5566953c', 'AssociationId': 'aclassoc-5c659635'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response, 'aclassoc-17b85d7e')