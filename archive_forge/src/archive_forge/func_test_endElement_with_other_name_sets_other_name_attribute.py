from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_endElement_with_other_name_sets_other_name_attribute(self):
    self.volume_attribute.endElement('someName', 'some value', None)
    self.assertEqual(self.volume_attribute.someName, 'some value')