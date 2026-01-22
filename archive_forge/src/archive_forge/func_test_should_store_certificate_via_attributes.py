from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_store_certificate_via_attributes(self):
    data = {'container_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    container = self.manager.create_certificate()
    container.name = self.container.name
    container.certificate = self.container.secret
    container.private_key = self.container.secret
    container.private_key_passphrase = self.container.secret
    container.intermediates = self.container.secret
    container_href = container.store()
    self.assertEqual(self.entity_href, container_href)
    self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
    container_req = jsonutils.loads(self.responses.last_request.text)
    self.assertEqual(self.container.name, container_req['name'])
    self.assertEqual('certificate', container_req['type'])
    self.assertCountEqual(self.container.certificate_secret_refs_json, container_req['secret_refs'])