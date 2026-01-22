import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
def test_image_list(self):
    images = self.mgr.list()
    expect = [('GET', '/v1/images/', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(images, matchers.HasLength(2))