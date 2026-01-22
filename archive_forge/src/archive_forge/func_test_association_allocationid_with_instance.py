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
def test_association_allocationid_with_instance(self):
    server = self.fc.servers.list()[0]
    self.patchobject(self.fc.servers, 'get', return_value=server)
    self.mock_show_network()
    self.mock_create_floatingip()
    self.mock_list_instance_ports()
    self.mock_no_router_for_vpc()
    t = template_format.parse(eip_template_ipassoc3)
    stack = utils.parse_stack(t)
    rsrc = self.create_eip(t, stack, 'the_eip')
    association = self.create_association(t, stack, 'IPAssoc')
    scheduler.TaskRunner(association.delete)()
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((association.DELETE, association.COMPLETE), association.state)
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)