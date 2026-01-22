from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
def test_delete_vpc_peering_connection(self):
    vpc_conn = VPCConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
    mock_response = mock.Mock()
    mock_response.read.return_value = self.DESCRIBE_VPC_PEERING_CONNECTIONS
    mock_response.status = 200
    vpc_conn.make_request = mock.Mock(return_value=mock_response)
    vpc_peering_connections = vpc_conn.get_all_vpc_peering_connections()
    self.assertEquals(1, len(vpc_peering_connections))
    vpc_peering_connection = vpc_peering_connections[0]
    mock_response = mock.Mock()
    mock_response.read.return_value = self.DELETE_VPC_PEERING_CONNECTION
    mock_response.status = 200
    vpc_conn.make_request = mock.Mock(return_value=mock_response)
    self.assertEquals(True, vpc_peering_connection.delete())
    self.assertIn('DeleteVpcPeeringConnection', vpc_conn.make_request.call_args_list[0][0])
    self.assertNotIn('DeleteVpc', vpc_conn.make_request.call_args_list[0][0])