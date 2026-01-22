import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_request_environment_info(self):
    result = self.beanstalk.request_environment_info(environment_name=self.environment, info_type='tail')
    self.assertIsInstance(result, response.RequestEnvironmentInfoResponse)
    self.wait_for_env(self.environment)
    result = self.beanstalk.retrieve_environment_info(environment_name=self.environment, info_type='tail')
    self.assertIsInstance(result, response.RetrieveEnvironmentInfoResponse)