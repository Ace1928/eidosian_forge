import copy
from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception as heat_ex
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.nova import floatingip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def prepare_floating_ip_assoc(self):
    return_server = self.novaclient.servers.list()[1]
    self.patchobject(self.novaclient.servers, 'get', return_value=return_server)
    iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
    self.patchobject(return_server, 'interface_list', return_value=[iface])
    template = template_format.parse(floating_ip_template_with_assoc)
    self.stack = utils.parse_stack(template)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    floating_ip_assoc = resource_defns['MyFloatingIPAssociation']
    return floatingip.NovaFloatingIpAssociation('MyFloatingIPAssociation', floating_ip_assoc, self.stack)