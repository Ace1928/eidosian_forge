import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def delete_elastic_ip(self, eip):
    new_eip = self.api.get_all_addresses([eip.public_ip])[0]
    if new_eip.association_id:
        new_eip.disassociate()
    new_eip.release()
    time.sleep(10)