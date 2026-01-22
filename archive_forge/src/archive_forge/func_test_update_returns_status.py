from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_update_returns_status(self):
    self.eni_one.connection = mock.Mock()
    self.eni_one.connection.get_all_network_interfaces.return_value = [self.eni_two]
    retval = self.eni_one.update()
    self.assertEqual(retval, 'two_status')