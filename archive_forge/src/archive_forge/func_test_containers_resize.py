import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_resize(self):
    containers = self.mgr.resize(CONTAINER1['id'], tty_width, tty_height)
    expects = []
    expects.append([('POST', '/v1/containers/%s/resize?w=%s&h=%s' % (CONTAINER1['id'], tty_width, tty_height), {'Content-Length': '0'}, None)])
    expects.append([('POST', '/v1/containers/%s/resize?h=%s&w=%s' % (CONTAINER1['id'], tty_height, tty_width), {'Content-Length': '0'}, None)])
    self.assertTrue(self.api.calls in expects)
    self.assertIsNone(containers)