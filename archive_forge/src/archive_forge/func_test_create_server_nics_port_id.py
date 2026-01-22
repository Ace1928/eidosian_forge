import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_nics_port_id(self):
    """Verify port-id in nics input turns into port in REST."""
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    active_server = fakes.make_fake_server('1234', '', 'BUILD')
    image_id = uuid.uuid4().hex
    port_id = uuid.uuid4().hex
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': fakes.FLAVOR_ID, 'imageRef': image_id, 'max_count': 1, 'min_count': 1, 'networks': [{'port': port_id}], 'name': 'server-name'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': active_server}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []})])
    self.cloud.create_server('server-name', dict(id=image_id), dict(id=fakes.FLAVOR_ID), nics=[{'port-id': port_id}], wait=False)
    self.assert_calls()