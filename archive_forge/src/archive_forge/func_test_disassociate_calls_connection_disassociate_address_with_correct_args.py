from tests.compat import mock, unittest
from boto.ec2.address import Address
def test_disassociate_calls_connection_disassociate_address_with_correct_args(self):
    self.address.disassociate()
    self.address.connection.disassociate_address.assert_called_with(public_ip='192.168.1.1', dry_run=False)