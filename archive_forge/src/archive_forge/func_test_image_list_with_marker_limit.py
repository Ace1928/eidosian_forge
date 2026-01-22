import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
def test_image_list_with_marker_limit(self):
    expect = [('GET', '/v1/images/?limit=2&marker=%s' % IMAGE2['image_id'], {}, None)]
    self._test_image_list_with_filters(limit=2, marker=IMAGE2['image_id'], expect=expect)