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
def mock_router_for_vpc(self):
    vpc_name = utils.PhysName('test_stack', 'the_vpc')
    self.mock_list_routers.return_value = {'routers': [{'status': 'ACTIVE', 'external_gateway_info': {'network_id': 'zzzz', 'enable_snat': True}, 'name': vpc_name, 'admin_state_up': True, 'tenant_id': '3e21026f2dc94372b105808c0e721661', 'routes': [], 'id': 'bbbb'}]}