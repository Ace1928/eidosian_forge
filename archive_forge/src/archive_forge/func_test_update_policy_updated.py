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
def test_update_policy_updated(self):
    updt_template = tmpl_with_updt_policy()
    grp = updt_template['resources']['group1']
    policy = grp['update_policy']['rolling_update']
    policy['min_in_service'] = '2'
    policy['max_batch_size'] = '4'
    policy['pause_time'] = '90'
    self.validate_update_policy_diff(tmpl_with_updt_policy(), updt_template)