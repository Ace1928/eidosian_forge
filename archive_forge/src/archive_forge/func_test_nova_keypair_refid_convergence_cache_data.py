import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_nova_keypair_refid_convergence_cache_data(self):
    cache_data = {'kp': {'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'}}
    stack = utils.parse_stack(self.kp_template, cache_data=cache_data)
    rsrc = stack.defn['kp']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())