import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
@classmethod
def wait_for_env(cls, env_name, status='Ready'):
    while not cls.env_ready(env_name, status):
        time.sleep(15)