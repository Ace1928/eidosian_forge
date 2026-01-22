import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import image_members
def test_update_image_members(self):
    image_id = IMAGE
    member_id = MEMBER
    status = 'accepted'
    image_member = self.controller.update(image_id, member_id, status)
    self.assertEqual(IMAGE, image_member.image_id)
    self.assertEqual(MEMBER, image_member.member_id)
    self.assertEqual(status, image_member.status)