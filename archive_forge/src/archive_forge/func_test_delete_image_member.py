import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import image_members
def test_delete_image_member(self):
    image_id = IMAGE
    member_id = MEMBER
    self.controller.delete(image_id, member_id)
    expect = [('DELETE', '/v2/images/{image}/members/{mem}'.format(image=IMAGE, mem=MEMBER), {}, None)]
    self.assertEqual(expect, self.api.calls)