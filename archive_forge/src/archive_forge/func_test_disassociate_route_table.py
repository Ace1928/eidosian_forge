from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_disassociate_route_table(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.disassociate_route_table('rtbassoc-fdad4894')
    self.assert_request_parameters({'Action': 'DisassociateRouteTable', 'AssociationId': 'rtbassoc-fdad4894'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)