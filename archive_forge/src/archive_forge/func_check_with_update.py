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
def check_with_update(self, with_policy=False, with_diff=False):
    current = copy.deepcopy(template)
    self.current_stack = utils.parse_stack(current)
    self.current_grp = self.current_stack['group1']
    current_grp_json = self.current_grp.frozen_definition()
    prop_diff, tmpl_diff = (None, None)
    updated = tmpl_with_updt_policy() if with_policy else copy.deepcopy(template)
    if with_diff:
        res_def = updated['resources']['group1']['properties']['resource_def']
        res_def['properties']['Foo'] = 'baz'
        prop_diff = dict({'count': 2, 'resource_def': {'properties': {'Foo': 'baz'}, 'type': 'OverwrittenFnGetRefIdType'}})
    updated_stack = utils.parse_stack(updated)
    updated_grp = updated_stack['group1']
    updated_grp_json = updated_grp.t.freeze()
    tmpl_diff = updated_grp.update_template_diff(updated_grp_json, current_grp_json)
    self.current_grp._replace = mock.Mock(return_value=[])
    self.current_grp._assemble_nested = mock.Mock()
    self.patchobject(scheduler.TaskRunner, 'start')
    self.current_grp.handle_update(updated_grp_json, tmpl_diff, prop_diff)