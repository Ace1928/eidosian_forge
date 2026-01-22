import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_network_list(self):
    networks = self.mgr.network_list(CONTAINER1['id'])
    expect = [('GET', '/v1/containers/%s/network_list' % CONTAINER1['id'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(networks)