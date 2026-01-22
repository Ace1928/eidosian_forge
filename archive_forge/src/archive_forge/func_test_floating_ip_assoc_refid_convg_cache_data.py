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
def test_floating_ip_assoc_refid_convg_cache_data(self):
    t = template_format.parse(floating_ip_template_with_assoc)
    cache_data = {'MyFloatingIPAssociation': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
    stack = utils.parse_stack(t, cache_data=cache_data)
    rsrc = stack.defn['MyFloatingIPAssociation']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())