from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_detach_with_no_attach_data(self):
    self.eni_two.connection = mock.Mock()
    self.eni_two.detach()
    self.eni_two.connection.detach_network_interface.assert_called_with(None, False, dry_run=False)