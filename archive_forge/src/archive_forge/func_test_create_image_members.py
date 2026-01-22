import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import image_members
def test_create_image_members(self):
    image_id = IMAGE
    member_id = MEMBER
    status = 'pending'
    image_member = self.controller.create(image_id, member_id)
    self.assertEqual(IMAGE, image_member.image_id)
    self.assertEqual(MEMBER, image_member.member_id)
    self.assertEqual(status, image_member.status)