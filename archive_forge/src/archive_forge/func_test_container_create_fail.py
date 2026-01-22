import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_container_create_fail(self):
    create_container_fail = copy.deepcopy(CREATE_CONTAINER1)
    create_container_fail['wrong_key'] = 'wrong'
    self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(containers.CREATION_ATTRIBUTES), self.mgr.create, **create_container_fail)
    self.assertEqual([], self.api.calls)