import testtools
from glanceclient.tests import utils
import glanceclient.v1.image_members
import glanceclient.v1.images
def test_list_by_image(self):
    members = self.mgr.list(image=self.image)
    expect = [('GET', '/v1/images/1/members', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(members))
    self.assertEqual('1', members[0].member_id)
    self.assertEqual('1', members[0].image_id)
    self.assertEqual(False, members[0].can_share)