import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def post_terminate_cleanup(self):
    """Helper to run clean up tasks after instances are removed."""
    for fn, args in self.post_terminate_cleanups:
        fn(*args)
        time.sleep(10)
    if self.vpc:
        self.api.delete_vpc(self.vpc.id)