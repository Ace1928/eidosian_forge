from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_openstack_cloud(self):
    self.assertIsInstance(self.cloud, connection.Connection)