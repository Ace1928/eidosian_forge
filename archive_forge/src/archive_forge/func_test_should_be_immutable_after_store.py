from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_be_immutable_after_store(self):
    data = {'container_ref': self.entity_href}
    self.responses.post(self.entity_base + '/', json=data)
    container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
    container_href = container.store()
    self.assertEqual(self.entity_href, container_href)
    attributes = ['name']
    for attr in attributes:
        try:
            setattr(container, attr, 'test')
            self.fail("didn't raise an ImmutableException exception")
        except base.ImmutableException:
            pass
    self.assertRaises(base.ImmutableException, container.add, self.container.secret.name, self.container.secret)