from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_get_generic_container(self, container_ref=None):
    container_ref = container_ref or self.entity_href
    data = self.container.get_dict(container_ref)
    self.responses.get(self.entity_href, json=data)
    container = self.manager.get(container_ref=container_ref)
    self.assertIsInstance(container, containers.Container)
    self.assertEqual(container_ref, container.container_ref)
    self.assertEqual(self.entity_href, self.responses.last_request.url)
    self.assertIsNotNone(container.secrets)