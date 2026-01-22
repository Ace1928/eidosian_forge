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
def test_floating_ip_assoc_create(self):
    rsrc = self.prepare_floating_ip_assoc()
    fip_request = {'floatingip': {'fixed_ip_address': '1.2.3.4', 'port_id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}}
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.mock_upd_fip.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip_request)
    self.assertIsNotNone(rsrc.id)
    self.assertEqual(rsrc.id, rsrc.resource_id)