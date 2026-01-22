import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_rebuild_environment(self):
    result = self.beanstalk.rebuild_environment(environment_name=self.environment)
    self.assertIsInstance(result, response.RebuildEnvironmentResponse)
    self.wait_for_env(self.environment)