import contextlib
import threading
from kazoo.protocol import paths as k_paths
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from zake import utils as zake_utils
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_zookeeper
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.types import entity
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def test_register_entity(self):
    conductor_name = 'conductor-abc@localhost:4123'
    entity_instance = entity.Entity('conductor', conductor_name, {})
    with base.connect_close(self.board):
        self.board.register_entity(entity_instance)
    self.assertIn(self.board.entity_path, self.client.storage.paths)
    conductor_entity_path = k_paths.join(self.board.entity_path, 'conductor', conductor_name)
    self.assertIn(conductor_entity_path, self.client.storage.paths)
    conductor_data = self.client.storage.paths[conductor_entity_path]['data']
    self.assertTrue(len(conductor_data) > 0)
    self.assertDictEqual({'name': conductor_name, 'kind': 'conductor', 'metadata': {}}, jsonutils.loads(misc.binary_decode(conductor_data)))
    entity_instance_2 = entity.Entity('non-sense', 'other_name', {})
    with base.connect_close(self.board):
        self.assertRaises(excp.NotImplementedError, self.board.register_entity, entity_instance_2)