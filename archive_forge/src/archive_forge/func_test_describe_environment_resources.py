import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_describe_environment_resources(self):
    result = self.beanstalk.describe_environment_resources(environment_name=self.environment)
    self.assertIsInstance(result, response.DescribeEnvironmentResourcesResponse)