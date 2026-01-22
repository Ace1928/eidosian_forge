from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_find_nova_addresses_key_name_and_ext_tag(self):
    addrs = {'public': [{'OS-EXT-IPS:type': 'fixed', 'addr': '198.51.100.2', 'version': 4}]}
    self.assertEqual(['198.51.100.2'], meta.find_nova_addresses(addrs, key_name='public', ext_tag='fixed'))
    self.assertEqual([], meta.find_nova_addresses(addrs, key_name='public', ext_tag='foo'))
    self.assertEqual([], meta.find_nova_addresses(addrs, key_name='bar', ext_tag='fixed'))