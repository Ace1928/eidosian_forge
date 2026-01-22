import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_router_interface_conflict(self):
    self.add_if_mock.side_effect = [qe.Conflict, None]
    t = template_format.parse(neutron_template)
    stack = utils.parse_stack(t)
    props = {'router': '3e46229d-8fce-4733-819a-b5fe630550f8', 'subnet': '91e47a57-7508-46fe-afc9-fc454e8580e1'}

    def find_rsrc(resource, name_or_id, cmd_resource=None):
        return props.get(resource, resource)
    self.find_rsrc_mock.side_effect = find_rsrc
    self.create_router_interface(t, stack, 'router_interface', properties=props)
    self.assertEqual(2, self.add_if_mock.call_count)