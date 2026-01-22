import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_create_storage_location(self):
    result = self.beanstalk.create_storage_location()
    self.assertIsInstance(result, response.CreateStorageLocationResponse)