from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import node_data
from heat.engine.resources.aws.iam import user
from heat.engine.resources.openstack.heat import access_policy as ap
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_accesspolicy_create_ok(self):
    t = template_format.parse(user_policy_template)
    stack = utils.parse_stack(t)
    resource_name = 'WebServerAccessPolicy'
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = ap.AccessPolicy(resource_name, resource_defns[resource_name], stack)
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)