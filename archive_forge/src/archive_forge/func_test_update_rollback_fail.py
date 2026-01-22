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
def test_update_rollback_fail(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'AParam': {'Type': 'String'}}, 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'abc'}}}}
    env1 = environment.Environment({'parameters': {'AParam': 'abc'}})
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl, env=env1), disable_rollback=False)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'BParam': {'Type': 'String'}}, 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'xyz'}}}}
    env2 = environment.Environment({'parameters': {'BParam': 'smelly'}})
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2, env=env2), disable_rollback=False)
    mock_create = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_create', side_effect=Exception)
    mock_delete = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_delete', side_effect=Exception)
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.ROLLBACK, stack.Stack.FAILED), self.stack.state)
    mock_create.assert_called_once_with()
    mock_delete.assert_called_once_with()