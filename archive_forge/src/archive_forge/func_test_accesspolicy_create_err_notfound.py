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
def test_accesspolicy_create_err_notfound(self):
    t = template_format.parse(user_policy_template)
    resource_name = 'WebServerAccessPolicy'
    t['Resources'][resource_name]['Properties']['AllowedResources'] = ['NoExistResource']
    stack = utils.parse_stack(t)
    self.assertRaises(exception.StackValidationFailed, stack.validate)