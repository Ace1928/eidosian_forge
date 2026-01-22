import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os.keystone import fake_keystoneclient
from heat.engine import environment
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_delete_stack_when_update_failed_twice(self):
    """Test when stack update failed twice and delete the stack.

        Test checks the following scenario:
        1. Create stack
        2. Update stack (failed)
        3. Update stack (failed)
        4. Delete stack
        The test checks the behavior of backup stack when update is failed.
        If some resources were not backed up correctly then test will fail.
        """
    tmpl_create = {'heat_template_version': '2013-05-23', 'resources': {'Ares': {'type': 'GenericResourceType'}}}
    self.stack = stack.Stack(self.ctx, 'update_fail_test_stack', template.Template(tmpl_create), disable_rollback=True)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    tmpl_update = {'heat_template_version': '2013-05-23', 'parameters': {'aparam': {'type': 'number', 'default': 1}}, 'resources': {'Ares': {'type': 'GenericResourceType'}, 'Bres': {'type': 'GenericResourceType'}, 'Cres': {'type': 'ResourceWithPropsRefPropOnDelete', 'properties': {'Foo': {'get_resource': 'Bres'}, 'FooInt': {'get_param': 'aparam'}}}}}
    mock_create = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_create', side_effect=[Exception, Exception])
    updated_stack_first = stack.Stack(self.ctx, 'update_fail_test_stack', template.Template(tmpl_update))
    self.stack.update(updated_stack_first)
    self.stack.resources['Cres'].resource_id_set('c_res')
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.FAILED), self.stack.state)
    updated_stack_second = stack.Stack(self.ctx, 'update_fail_test_stack', template.Template(tmpl_update))
    self.stack.update(updated_stack_second)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.FAILED), self.stack.state)
    self.assertEqual(mock_create.call_count, 2)
    self.stack.delete()
    self.assertEqual((stack.Stack.DELETE, stack.Stack.COMPLETE), self.stack.state)