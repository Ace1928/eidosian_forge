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
def test_show_snapshot_not_belong_to_stack(self):
    stk1 = self._create_stack('stack_snaphot_not_belong_to_stack_1')
    stk1._persist_state()
    snapshot1 = self.engine.stack_snapshot(self.ctx, stk1.identifier(), 'snap1')
    snapshot_id = snapshot1['id']
    stk2 = self._create_stack('stack_snaphot_not_belong_to_stack_2')
    stk2._persist_state()
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.show_snapshot, self.ctx, stk2.identifier(), snapshot_id)
    expected = 'The Snapshot (%(snapshot)s) for Stack (%(stack)s) could not be found' % {'snapshot': snapshot_id, 'stack': stk2.name}
    self.assertEqual(exception.SnapshotNotFound, ex.exc_info[0])
    self.assertIn(expected, str(ex.exc_info[1]))