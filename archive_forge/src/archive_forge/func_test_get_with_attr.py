from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
def test_get_with_attr(self):
    fields = {'stack_id': 'teststack', 'resource_name': 'testresource', 'with_attr': ['attr_a', 'attr_b']}
    expect = ('GET', '/stacks/teststack/abcd1234/resources/testresource?with_attr=attr_a&with_attr=attr_b')
    key = 'resource'
    self._base_test('get', fields, expect, key)