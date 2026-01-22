import boto
from boto.route53.domains.exceptions import InvalidInput
from tests.compat import unittest
def test_handle_invalid_input_error(self):
    with self.assertRaises(InvalidInput):
        self.route53domains.check_domain_availability(domain_name='!amazon.com')