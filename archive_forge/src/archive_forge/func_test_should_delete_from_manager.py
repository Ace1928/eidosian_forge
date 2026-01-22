from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_delete_from_manager(self, container_ref=None):
    container_ref = container_ref or self.entity_href
    self.responses.delete(self.entity_href, status_code=204)
    self.manager.delete(container_ref=container_ref)
    self.assertEqual(self.entity_href, self.responses.last_request.url)