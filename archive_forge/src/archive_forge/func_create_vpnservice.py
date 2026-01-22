import copy
from unittest import mock
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import vpnservice
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def create_vpnservice(self, resolve_neutron=True, resolve_router=True):
    self.stub_SubnetConstraint_validate()
    self.stub_RouterConstraint_validate()
    if resolve_neutron:
        snippet = template_format.parse(vpnservice_template)
    else:
        snippet = template_format.parse(vpnservice_template_deprecated)
    if resolve_router:
        props = snippet['resources']['VPNService']['properties']
        props['router'] = 'rou123'
        del props['router_id']
    self.mockclient.create_vpnservice.return_value = {'vpnservice': {'id': 'vpn123'}}
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    return vpnservice.VPNService('vpnservice', resource_defns['VPNService'], self.stack)