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
def test_update_association_needs_update_InstanceId_EIP(self):
    server = self.fc.servers.list()[0]
    self.patchobject(self.fc.servers, 'get', return_value=server)
    iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
    self.patchobject(server, 'interface_list', return_value=[iface])
    self.mock_list_floatingips()
    self.mock_create_floatingip()
    t = template_format.parse(eip_template_ipassoc)
    stack = utils.parse_stack(t)
    self.create_eip(t, stack, 'IPAddress')
    after_props = {'InstanceId': '5678', 'EIP': '11.0.0.2'}
    before = self.create_association(t, stack, 'IPAssoc')
    after = rsrc_defn.ResourceDefinition(before.name, before.type(), after_props)
    updater = scheduler.TaskRunner(before.update, after)
    self.assertRaises(resource.UpdateReplace, updater)