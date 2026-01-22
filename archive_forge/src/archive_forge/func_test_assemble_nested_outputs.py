import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_assemble_nested_outputs(self):
    """Tests nested stack creation based on props.

        Tests that the nested stack that implements the group is created
        appropriately based on properties.
        """
    stack = utils.parse_stack(template)
    snip = stack.t.resource_definitions(stack)['group1']
    resg = resource_group.ResourceGroup('test', snip, stack)
    templ = {'heat_template_version': '2015-04-30', 'resources': {'0': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}, '1': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}, '2': {'type': 'OverwrittenFnGetRefIdType', 'properties': {'Foo': 'Bar'}}}, 'outputs': {'refs_map': {'value': {'0': {'get_resource': '0'}, '1': {'get_resource': '1'}, '2': {'get_resource': '2'}}}, 'foo': {'value': [{'get_attr': ['0', 'foo']}, {'get_attr': ['1', 'foo']}, {'get_attr': ['2', 'foo']}]}}}
    resg.referenced_attrs = mock.Mock(return_value=['foo'])
    self.assertEqual(templ, resg._assemble_nested(['0', '1', '2']).t)