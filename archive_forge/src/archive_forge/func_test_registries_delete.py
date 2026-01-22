import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registries_delete(self):
    registries = self.mgr.delete(REGISTRY1['uuid'])
    expect = [('DELETE', '/v1/registries/%s' % REGISTRY1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(registries)