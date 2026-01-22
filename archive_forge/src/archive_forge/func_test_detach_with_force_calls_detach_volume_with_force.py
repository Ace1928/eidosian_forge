from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_detach_with_force_calls_detach_volume_with_force(self):
    self.volume_one.connection = mock.Mock()
    self.volume_one.detach(True)
    self.volume_one.connection.detach_volume.assert_called_with(1, 2, '/dev/null', True, dry_run=False)