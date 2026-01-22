import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import services
def test_service_list_with_sort_key(self):
    expect = [('GET', '/v1/services/?sort_key=id', {}, None)]
    self._test_service_list_with_filters(sort_key='id', expect=expect)