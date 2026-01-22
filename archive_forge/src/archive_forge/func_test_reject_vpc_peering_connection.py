from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
def test_reject_vpc_peering_connection(self):
    self.set_http_response(status_code=200)
    self.assertEquals(self.service_connection.reject_vpc_peering_connection('pcx-12345678'), True)