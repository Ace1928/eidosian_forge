from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_volume_state_returns_status(self):
    retval = self.volume_one.volume_state()
    self.assertEqual(retval, 'one_status')