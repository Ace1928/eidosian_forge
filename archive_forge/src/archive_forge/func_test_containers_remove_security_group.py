import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_remove_security_group(self):
    containers = self.mgr.remove_security_group(CONTAINER1['id'], security_group)
    expect = [('POST', '/v1/containers/%s/remove_security_group?%s' % (CONTAINER1['id'], parse.urlencode({'name': security_group})), {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(containers)