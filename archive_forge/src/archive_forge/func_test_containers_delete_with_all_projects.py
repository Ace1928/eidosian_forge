import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_delete_with_all_projects(self):
    containers = self.mgr.delete(CONTAINER1['id'], all_projects=all_projects)
    expect = [('DELETE', '/v1/containers/%s?all_projects=%s' % (CONTAINER1['id'], all_projects), {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(containers)