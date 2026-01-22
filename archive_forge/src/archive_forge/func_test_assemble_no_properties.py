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
def test_assemble_no_properties(self):
    templ = copy.deepcopy(template)
    res_def = templ['resources']['group1']['properties']['resource_def']
    del res_def['properties']
    stack = utils.parse_stack(templ)
    resg = stack.resources['group1']
    self.assertIsNone(resg.validate())