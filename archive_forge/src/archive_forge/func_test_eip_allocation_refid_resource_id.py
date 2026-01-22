import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_eip_allocation_refid_resource_id(self):
    t = template_format.parse(eip_template_ipassoc)
    stack = utils.parse_stack(t)
    rsrc = stack['IPAssoc']
    rsrc.resource_id = 'phy-rsrc-id'
    self.assertEqual('phy-rsrc-id', rsrc.FnGetRefId())