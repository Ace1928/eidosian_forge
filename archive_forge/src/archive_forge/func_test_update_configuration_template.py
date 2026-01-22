import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_update_configuration_template(self):
    result = self.beanstalk.update_configuration_template(application_name=self.app_name, template_name=self.template)
    self.assertIsInstance(result, response.UpdateConfigurationTemplateResponse)