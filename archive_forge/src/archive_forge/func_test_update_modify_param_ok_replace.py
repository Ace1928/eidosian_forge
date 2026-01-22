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
def test_update_modify_param_ok_replace(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'foo': {'Type': 'String'}}, 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'foo'}}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl, env=environment.Environment({'foo': 'abc'})))
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    env2 = environment.Environment({'foo': 'xyz'})
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl, env=env2))

    def check_and_raise(*args):
        self.assertEqual('abc', self.stack['AResource']._stored_properties_data['Foo'])
        raise resource.UpdateReplace
    mock_upd = self.patchobject(generic_rsrc.ResourceWithProps, 'update_template_diff', side_effect=check_and_raise)
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('xyz', self.stack['AResource']._stored_properties_data['Foo'])
    after = rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'xyz'})
    before = rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'abc'})
    mock_upd.assert_called_once_with(after, before)