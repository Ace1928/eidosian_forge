import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import services
def test_service_list_with_sort_dir(self):
    expect = [('GET', '/v1/services/?sort_dir=asc', {}, None)]
    self._test_service_list_with_filters(sort_dir='asc', expect=expect)