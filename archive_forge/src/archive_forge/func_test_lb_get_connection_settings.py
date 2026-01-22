from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_lb_get_connection_settings(self):
    """Tests checking connectionSettings attribute"""
    mock_response, elb, _ = self._setup_mock()
    attrs = [('idle_timeout', 30)]
    mock_response.read.return_value = ATTRIBUTE_GET_CS_RESPONSE
    attributes = elb.get_all_lb_attributes('test_elb')
    self.assertTrue(isinstance(attributes, LbAttributes))
    for attr, value in attrs:
        self.assertEqual(getattr(attributes.connecting_settings, attr), value)