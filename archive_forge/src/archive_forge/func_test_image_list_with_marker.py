import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
def test_image_list_with_marker(self):
    expect = [('GET', '/v1/images/?marker=%s' % IMAGE2['image_id'], {}, None)]
    self._test_image_list_with_filters(marker=IMAGE2['image_id'], expect=expect)