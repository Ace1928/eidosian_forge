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
def test_update_replace_by_reference(self):
    """Test case for changes in dynamic attributes.

        Changes in dynamic attributes, due to other resources been updated
        are not ignored and can cause dependent resources to be updated.
        """
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'abc'}}, 'BResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'AResource'}}}}}
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'smelly'}}, 'BResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'AResource'}}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl))
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('abc', self.stack['AResource']._stored_properties_data['Foo'])
    self.assertEqual('AResource', self.stack['BResource']._stored_properties_data['Foo'])
    mock_id = self.patchobject(generic_rsrc.ResourceWithProps, 'get_reference_id', return_value='inst-007')
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2))
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('smelly', self.stack['AResource']._stored_properties_data['Foo'])
    self.assertEqual('inst-007', self.stack['BResource']._stored_properties_data['Foo'])
    mock_id.assert_called_with()