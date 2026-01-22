from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_cant_use_public_ip(self):
    collection = NetworkInterfaceCollection(self.network_interfaces_spec3, self.network_interfaces_spec1)
    params = {}
    with self.assertRaises(BotoClientError):
        collection.build_list_params(params, prefix='LaunchSpecification.')
    self.network_interfaces_spec3.device_index = 1
    collection = NetworkInterfaceCollection(self.network_interfaces_spec3)
    params = {}
    with self.assertRaises(BotoClientError):
        collection.build_list_params(params, prefix='LaunchSpecification.')