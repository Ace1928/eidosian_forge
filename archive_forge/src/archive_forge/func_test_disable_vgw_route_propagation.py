from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
def test_disable_vgw_route_propagation(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.disable_vgw_route_propagation('rtb-c98a35a0', 'vgw-d8e09e8a')
    self.assert_request_parameters({'Action': 'DisableVgwRoutePropagation', 'GatewayId': 'vgw-d8e09e8a', 'RouteTableId': 'rtb-c98a35a0'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(api_response, True)