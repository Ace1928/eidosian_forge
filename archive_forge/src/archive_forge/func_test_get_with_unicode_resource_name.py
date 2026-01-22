from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
def test_get_with_unicode_resource_name(self):
    fields = {'stack_id': 'teststack', 'resource_name': '工作'}
    expect = ('GET', '/stacks/teststack/abcd1234/resources/%E5%B7%A5%E4%BD%9C')
    key = 'resource'
    self._base_test('get', fields, expect, key)