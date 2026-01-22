import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def terminate_instances(self):
    """Helper to remove all instances and kick off additional cleanup
        once they are terminated.
        """
    for instance in self.instances:
        self.terminate_instance(instance)
    self.post_terminate_cleanup()