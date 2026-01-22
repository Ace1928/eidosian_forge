import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_name_with_dashes(self):
    conn = self._get_conn()
    discovery = {'versions': {'values': [{'status': 'stable', 'id': 'v1', 'links': [{'href': 'https://example.org:5050/v1', 'rel': 'self'}]}]}}
    status = {'finished': True, 'error': None}
    self.register_uris([dict(method='GET', uri='https://example.org:5050', json=discovery), dict(method='GET', uri='https://example.org:5050/v1', json=discovery), dict(method='GET', uri='https://example.org:5050/v1/introspection/abcd', json=status)])
    adap = conn.baremetal_introspection
    self.assertEqual('baremetal-introspection', adap.service_type)
    self.assertEqual('public', adap.interface)
    self.assertEqual('https://example.org:5050/v1', adap.endpoint_override)
    self.assertTrue(adap.get_introspection('abcd').is_finished)