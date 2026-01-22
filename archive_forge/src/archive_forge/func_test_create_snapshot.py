from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.objects import snapshot as snapshot_objects
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(stack.Stack, 'load')
def test_create_snapshot(self, mock_load):
    files = {'a_file': 'the contents'}
    stk = self._create_stack('stack_snapshot_create', files=files)
    mock_load.return_value = stk
    snapshot = self.engine.stack_snapshot(self.ctx, stk.identifier(), 'snap1')
    self.assertIsNotNone(snapshot['id'])
    self.assertIsNotNone(snapshot['creation_time'])
    self.assertEqual('snap1', snapshot['name'])
    self.assertEqual('IN_PROGRESS', snapshot['status'])
    snapshot = self.engine.show_snapshot(self.ctx, stk.identifier(), snapshot['id'])
    self.assertEqual('COMPLETE', snapshot['status'])
    self.assertEqual('SNAPSHOT', snapshot['data']['action'])
    self.assertEqual('COMPLETE', snapshot['data']['status'])
    self.assertEqual(files, snapshot['data']['files'])
    self.assertEqual(stk.id, snapshot['data']['id'])
    self.assertIsNotNone(stk.updated_time)
    self.assertIsNotNone(snapshot['creation_time'])
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)