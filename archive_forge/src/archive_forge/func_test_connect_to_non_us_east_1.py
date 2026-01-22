import boto
from boto.configservice.exceptions import NoSuchConfigurationRecorderException
from tests.compat import unittest
def test_connect_to_non_us_east_1(self):
    self.configservice = boto.configservice.connect_to_region('us-west-2')
    response = self.configservice.describe_configuration_recorders()
    self.assertIn('ConfigurationRecorders', response)