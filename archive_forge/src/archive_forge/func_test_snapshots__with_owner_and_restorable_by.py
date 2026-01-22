from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_snapshots__with_owner_and_restorable_by(self):
    self.volume_one.connection = mock.Mock()
    self.volume_one.connection.get_all_snapshots.return_value = []
    self.volume_one.snapshots('owner', 'restorable_by')
    self.volume_one.connection.get_all_snapshots.assert_called_with(owner='owner', restorable_by='restorable_by', dry_run=False)