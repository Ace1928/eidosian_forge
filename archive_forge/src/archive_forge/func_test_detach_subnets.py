from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
def test_detach_subnets(self):
    elb = ELBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
    lb = LoadBalancer(elb, 'mylb')
    mock_response = mock.Mock()
    mock_response.read.return_value = DETACH_RESPONSE
    mock_response.status = 200
    elb.make_request = mock.Mock(return_value=mock_response)
    lb.detach_subnets('s-xxx')