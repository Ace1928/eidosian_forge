import testtools
from glanceclient.tests import utils
import glanceclient.v1.image_members
import glanceclient.v1.images
def test_list_by_member(self):
    resource_class = glanceclient.v1.image_members.ImageMember
    member = resource_class(self.api, {'member_id': '1'}, True)
    self.mgr.list(member=member)
    expect = [('GET', '/v1/shared-images/1', {}, None)]
    self.assertEqual(expect, self.api.calls)