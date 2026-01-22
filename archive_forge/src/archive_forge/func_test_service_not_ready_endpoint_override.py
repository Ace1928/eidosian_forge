import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_service_not_ready_endpoint_override(self):
    conn = self._get_conn()
    discovery = {'versions': {'values': [{'status': 'stable', 'id': 'v1', 'links': [{'href': 'https://example.org:5050/v1', 'rel': 'self'}]}]}}
    status = {'finished': True, 'error': None}
    self.register_uris([dict(method='GET', uri='https://example.org:5050', exc=requests.exceptions.ConnectTimeout), dict(method='GET', uri='https://example.org:5050', json=discovery), dict(method='GET', uri='https://example.org:5050/v1', json=discovery), dict(method='GET', uri='https://example.org:5050/v1/introspection/abcd', json=status)])
    self.assertRaises(exceptions.ServiceDiscoveryException, getattr, conn, 'baremetal_introspection')
    adap = conn.baremetal_introspection
    self.assertEqual('baremetal-introspection', adap.service_type)
    self.assertEqual('public', adap.interface)
    self.assertEqual('https://example.org:5050/v1', adap.endpoint_override)
    self.assertTrue(adap.get_introspection('abcd').is_finished)