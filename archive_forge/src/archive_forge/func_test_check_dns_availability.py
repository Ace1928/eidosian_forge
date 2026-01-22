import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_check_dns_availability(self):
    result = self.beanstalk.check_dns_availability('amazon')
    self.assertIsInstance(result, response.CheckDNSAvailabilityResponse, 'incorrect response object returned')
    self.assertFalse(result.available)