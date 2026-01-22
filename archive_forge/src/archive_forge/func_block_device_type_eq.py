from tests.compat import unittest
from boto.ec2.connection import EC2Connection
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from tests.compat import OrderedDict
from tests.unit import AWSMockServiceTestCase
def block_device_type_eq(self, b1, b2):
    if isinstance(b1, BlockDeviceType) and isinstance(b2, BlockDeviceType):
        return all([b1.connection == b2.connection, b1.ephemeral_name == b2.ephemeral_name, b1.no_device == b2.no_device, b1.volume_id == b2.volume_id, b1.snapshot_id == b2.snapshot_id, b1.status == b2.status, b1.attach_time == b2.attach_time, b1.delete_on_termination == b2.delete_on_termination, b1.size == b2.size, b1.encrypted == b2.encrypted])