from tests.compat import unittest
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from tests.compat import OrderedDict
from tests.unit import AWSMockServiceTestCase
def test_endElement_with_name_deleteOnTermination_value_other(self):
    self.block_device_type.endElement('deleteOnTermination', 'something else', None)
    self.assertEqual(self.block_device_type.delete_on_termination, False)