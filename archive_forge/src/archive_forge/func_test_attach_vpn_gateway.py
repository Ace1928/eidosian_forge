from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
def test_attach_vpn_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.attach_vpn_gateway('vgw-8db04f81', 'vpc-1a2b3c4d')
    self.assert_request_parameters({'Action': 'AttachVpnGateway', 'VpnGatewayId': 'vgw-8db04f81', 'VpcId': 'vpc-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, Attachment)
    self.assertEquals(api_response.vpc_id, 'vpc-1a2b3c4d')
    self.assertEquals(api_response.state, 'attaching')