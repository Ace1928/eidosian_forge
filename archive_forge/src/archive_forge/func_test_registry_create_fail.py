import copy
import testtools
from testtools import matchers
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import registries
def test_registry_create_fail(self):
    create_registry_fail = copy.deepcopy(CREATE_REGISTRY1)
    create_registry_fail['wrong_key'] = 'wrong'
    self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(registries.CREATION_ATTRIBUTES), self.mgr.create, **create_registry_fail)
    self.assertEqual([], self.api.calls)