import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_network_attach(self):
    containers = self.mgr.network_attach(CONTAINER1['id'], network='neutron_network')
    expect = [('POST', '/v1/containers/%s/network_attach?%s' % (CONTAINER1['id'], parse.urlencode({'network': 'neutron_network'})), {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(containers)