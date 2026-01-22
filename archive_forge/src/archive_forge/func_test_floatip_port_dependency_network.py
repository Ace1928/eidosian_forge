import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import neutron
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_floatip_port_dependency_network(self):
    t = template_format.parse(neutron_floating_no_assoc_template)
    del t['resources']['port_floating']['properties']['fixed_ips']
    stack = utils.parse_stack(t)
    p_show = self.mockclient.show_network
    p_show.return_value = {'network': {'subnets': ['subnet_uuid']}}
    p_result = self.patchobject(hot_funcs.GetResource, 'result', autospec=True)

    def return_uuid(self):
        if self.args == 'network':
            return 'net_uuid'
        return 'subnet_uuid'
    p_result.side_effect = return_uuid
    required_by = set(stack.dependencies.required_by(stack['router_interface']))
    self.assertIn(stack['floating_ip'], required_by)
    p_show.assert_called_once_with('net_uuid')