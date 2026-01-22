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
def test_floating_ip_assoc_update_fl_ip(self):
    rsrc = self.prepare_floating_ip_assoc()
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    props = copy.deepcopy(rsrc.properties.data)
    props['floating_ip'] = 'fc68ea2c-cccc-4b4f-bd82-94ec81110766'
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, update_snippet)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    fip_request = {'floatingip': {'fixed_ip_address': '1.2.3.4', 'port_id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}}
    fip_request_none = {'floatingip': {'fixed_ip_address': None, 'port_id': None}}
    calls = [mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip_request), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip_request_none), mock.call('fc68ea2c-cccc-4b4f-bd82-94ec81110766', fip_request)]
    self.mock_upd_fip.assert_has_calls(calls)
    self.assertEqual(3, self.mock_upd_fip.call_count)