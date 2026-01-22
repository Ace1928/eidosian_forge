from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_startElement_with_name_autoEnableIO_sets_key_name(self):
    self.volume_attribute.startElement('autoEnableIO', None, None)
    self.assertEqual(self.volume_attribute._key_name, 'autoEnableIO')