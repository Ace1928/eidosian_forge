from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_attachment_state_returns_state(self):
    retval = self.volume_one.attachment_state()
    self.assertEqual(retval, 'some status')