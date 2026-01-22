import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_restart(self):
    containers = self.mgr.restart(CONTAINER1['id'], timeout)
    expect = [('POST', '/v1/containers/%s/reboot?timeout=10' % CONTAINER1['id'], {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(containers)