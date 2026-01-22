from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_delete_route(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_route('rtb-e4ad488d', '172.16.1.0/24')
    self.assert_request_parameters({'Action': 'DeleteRoute', 'RouteTableId': 'rtb-e4ad488d', 'DestinationCidrBlock': '172.16.1.0/24'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)