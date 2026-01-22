import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_coontainer_list_with_sort_dir(self):
    expect = [('GET', '/v1/registries/?sort_dir=asc', {}, None)]
    self._test_registries_list_with_filters(sort_dir='asc', expect=expect)