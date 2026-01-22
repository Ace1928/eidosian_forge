import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
def test_image_list_with_sort_key_dir(self):
    expect = [('GET', '/v1/images/?sort_key=image_id&sort_dir=desc', {}, None)]
    self._test_image_list_with_filters(sort_key='image_id', sort_dir='desc', expect=expect)