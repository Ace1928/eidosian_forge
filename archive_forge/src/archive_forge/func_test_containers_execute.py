import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_execute(self):
    containers = self.mgr.execute(CONTAINER1['id'], command=CONTAINER1['command'])
    expect = [('POST', '/v1/containers/%s/execute?%s' % (CONTAINER1['id'], parse.urlencode({'command': CONTAINER1['command']})), {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(containers)