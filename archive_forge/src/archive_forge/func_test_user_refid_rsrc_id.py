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
def test_user_refid_rsrc_id(self):
    t = template_format.parse(user_template)
    stack = utils.parse_stack(t)
    rsrc = stack['CfnUser']
    rsrc.resource_id = 'phy-rsrc-id'
    self.assertEqual('phy-rsrc-id', rsrc.FnGetRefId())