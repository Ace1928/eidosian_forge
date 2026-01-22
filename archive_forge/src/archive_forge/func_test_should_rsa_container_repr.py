from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
def test_should_rsa_container_repr(self):
    container_obj = self.manager.create_rsa(name=self.container.name)
    self.assertIn('name="{0}"'.format(self.container.name), repr(container_obj))