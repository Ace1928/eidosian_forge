from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.resources.openstack.neutron import network_gateway
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def mock_create_fail_network_not_found_delete_success(self):
    self.mockclient.create_network_gateway.return_value = {'network_gateway': {'id': 'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', 'name': 'NetworkGateway', 'default': False, 'tenant_id': '96ba52dc-c5c5-44c6-9a9d-d3ba1a03f77f', 'devices': [{'id': 'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': 'breth1'}]}}
    self.mockclient.connect_network_gateway.side_effect = qe.NeutronClientException
    self.mockclient.disconnect_network_gateway.return_value = None
    self.mockclient.delete_network_gateway.return_value = None
    self.mockclient.show_network_gateway.side_effect = qe.NeutronClientException(status_code=404)
    t = template_format.parse(gw_template)
    self.stack = utils.parse_stack(t)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    rsrc = network_gateway.NetworkGateway('test_network_gateway', resource_defns['NetworkGateway'], self.stack)
    return rsrc