import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_12_describe_environments(self):
    result = self.beanstalk.describe_environments()
    self.assertIsInstance(result, response.DescribeEnvironmentsResponse)