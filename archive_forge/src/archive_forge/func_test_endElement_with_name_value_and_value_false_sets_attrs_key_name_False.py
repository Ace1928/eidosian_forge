from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_endElement_with_name_value_and_value_false_sets_attrs_key_name_False(self):
    self.volume_attribute._key_name = 'other_key_name'
    self.volume_attribute.endElement('value', 'false', None)
    self.assertEqual(self.volume_attribute.attrs['other_key_name'], False)