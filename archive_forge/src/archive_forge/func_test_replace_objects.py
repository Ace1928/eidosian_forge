import testtools
from glanceclient.tests import utils
import glanceclient.v1.image_members
import glanceclient.v1.images
def test_replace_objects(self):
    body = [glanceclient.v1.image_members.ImageMember(self.mgr, {'member_id': '2', 'can_share': False}, True), glanceclient.v1.image_members.ImageMember(self.mgr, {'member_id': '3', 'can_share': True}, True)]
    self.mgr.replace(self.image, body)
    expect_body = {'memberships': [{'member_id': '2', 'can_share': False}, {'member_id': '3', 'can_share': True}]}
    expect = [('PUT', '/v1/images/1/members', {}, sorted(expect_body.items()))]
    self.assertEqual(expect, self.api.calls)