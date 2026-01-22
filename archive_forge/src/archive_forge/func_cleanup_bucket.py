import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def cleanup_bucket(self, bucket):
    for key in bucket.get_all_keys():
        key.delete()
    bucket.delete()