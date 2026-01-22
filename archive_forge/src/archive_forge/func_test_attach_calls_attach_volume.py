from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_attach_calls_attach_volume(self):
    self.volume_one.connection = mock.Mock()
    self.volume_one.attach('instance_id', '/dev/null')
    self.volume_one.connection.attach_volume.assert_called_with(1, 'instance_id', '/dev/null', dry_run=False)