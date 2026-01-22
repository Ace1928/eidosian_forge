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
def test_show_snapshot_not_found(self):
    stk = self._create_stack('stack_snapshot_not_found')
    snapshot_id = str(uuid.uuid4())
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.show_snapshot, self.ctx, stk.identifier(), snapshot_id)
    expected = 'Snapshot with id %s not found' % snapshot_id
    self.assertEqual(exception.NotFound, ex.exc_info[0])
    self.assertIn(expected, str(ex.exc_info[1]))