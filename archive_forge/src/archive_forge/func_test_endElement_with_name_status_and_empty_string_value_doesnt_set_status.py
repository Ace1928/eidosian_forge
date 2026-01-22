from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_endElement_with_name_status_and_empty_string_value_doesnt_set_status(self):
    volume = Volume()
    volume.endElement('status', '', None)
    self.assertNotEqual(volume.status, '')