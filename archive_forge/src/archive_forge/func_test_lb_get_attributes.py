from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_lb_get_attributes(self):
    """Tests the LbAttributes from the ELB object."""
    mock_response, _, lb = self._setup_mock()
    for response, attr_tests in ATTRIBUTE_TESTS:
        mock_response.read.return_value = response
        attributes = lb.get_attributes(force=True)
        self.assertTrue(isinstance(attributes, LbAttributes))
        self._verify_attributes(attributes, attr_tests)