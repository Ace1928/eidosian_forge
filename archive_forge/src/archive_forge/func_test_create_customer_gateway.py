from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, CustomerGateway
def test_create_customer_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_customer_gateway('ipsec.1', '12.1.2.3', 65534)
    self.assert_request_parameters({'Action': 'CreateCustomerGateway', 'Type': 'ipsec.1', 'IpAddress': '12.1.2.3', 'BgpAsn': 65534}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, CustomerGateway)
    self.assertEquals(api_response.id, 'cgw-b4dc3961')
    self.assertEquals(api_response.state, 'pending')
    self.assertEquals(api_response.type, 'ipsec.1')
    self.assertEquals(api_response.ip_address, '12.1.2.3')
    self.assertEquals(api_response.bgp_asn, 65534)