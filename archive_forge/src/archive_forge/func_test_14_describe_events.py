import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_14_describe_events(self):
    result = self.beanstalk.describe_events()
    self.assertIsInstance(result, response.DescribeEventsResponse)