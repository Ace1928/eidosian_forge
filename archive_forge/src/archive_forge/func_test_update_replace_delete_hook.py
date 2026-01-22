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
def test_update_replace_delete_hook(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'foo': {'Type': 'String'}}, 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'foo'}}}}}
    env = environment.Environment({'foo': 'abc'})
    env.registry.load({'resources': {'AResource': {'hooks': 'pre-delete'}}})
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl, env=env))
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    env2 = environment.Environment({'foo': 'xyz'})
    env2.registry.load({'resources': {'AResource': {'hooks': 'pre-delete'}}})
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl, env=env2))
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('xyz', self.stack['AResource']._stored_properties_data['Foo'])