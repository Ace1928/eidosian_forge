import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def create_application(self):
    self.beanstalk.create_application(application_name=self.app_name)
    self.addCleanup(partial(self.beanstalk.delete_application, application_name=self.app_name))