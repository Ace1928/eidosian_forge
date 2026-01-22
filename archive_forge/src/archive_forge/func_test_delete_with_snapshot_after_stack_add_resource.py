import copy
import time
from unittest import mock
import fixtures
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_delete_with_snapshot_after_stack_add_resource(self):
    tpl = {'heat_template_version': 'queens', 'resources': {'A': {'type': 'ResourceWithRestoreType'}}}
    self.stack = stack.Stack(self.ctx, 'stack_delete_with_snapshot', template.Template(tpl))
    stack_id = self.stack.store()
    self.stack.create()
    data = copy.deepcopy(self.stack.prepare_abandon())
    data['resources']['A']['resource_data']['a_string'] = 'foo'
    snapshot_fake = {'tenant': self.ctx.tenant_id, 'name': 'Snapshot', 'stack_id': stack_id, 'status': 'COMPLETE', 'data': data}
    snapshot_object.Snapshot.create(self.ctx, snapshot_fake)
    self.assertIsNotNone(snapshot_object.Snapshot.get_all_by_stack(self.ctx, stack_id))
    new_tmpl = {'heat_template_version': 'queens', 'resources': {'A': {'type': 'ResourceWithRestoreType'}, 'B': {'type': 'ResourceWithRestoreType'}}}
    updated_stack = stack.Stack(self.ctx, 'update_stack_add_res', template.Template(new_tmpl))
    self.stack.update(updated_stack)
    self.assertEqual(2, len(self.stack.resources))
    self.stack.delete()
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNone(db_s)
    self.assertEqual((stack.Stack.DELETE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual([], snapshot_object.Snapshot.get_all_by_stack(self.ctx, stack_id))